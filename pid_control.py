import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import RPi.GPIO as GPIO
import threading
import atexit
import time
import torch.nn.functional as F

# 모터 핀 설정
MOTOR_PINS = {
    'left': {'A1': 20, 'B1': 21},
    'right': {'A1': 5, 'B1': 6}
}

GPIO.setmode(GPIO.BCM)
for motor in MOTOR_PINS.values():
    for pin in motor.values():
        GPIO.setup(pin, GPIO.OUT)

MOTOR_PWM = {}
for motor_label, motor in MOTOR_PINS.items():
    MOTOR_PWM[motor_label] = {
        'A1': GPIO.PWM(motor['A1'], 1000),
        'B1': GPIO.PWM(motor['B1'], 1000)
    }
    MOTOR_PWM[motor_label]['A1'].start(0)
    MOTOR_PWM[motor_label]['B1'].start(0)

class PIDController:
    def __init__(self, kp, ki, kd, log=False):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.previous_error = 0
        self.log = log

    def update(self, error, dt):
        self.integral += error * dt

        derivative = (error - self.previous_error) / dt

        output = self.kp * error + self.ki * self.integral + self.kd * derivative

        self.previous_error = error

        if self.log:
            print(f"Error: {error}, Integral: {self.integral}, Derivative: {derivative}, Output: {output}")

        return output


def set_motor_speed(motor_label, speed):
    # 속도 값을 0과 100 사이로 제한
    speed = max(0, min(100, speed))

    pwm_a = MOTOR_PWM[motor_label]['A1']
    pwm_b = MOTOR_PWM[motor_label]['B1']
    if speed >= 0:
        pwm_a.ChangeDutyCycle(speed)
        pwm_b.ChangeDutyCycle(0)
    else:
        pwm_a.ChangeDutyCycle(0)
        pwm_b.ChangeDutyCycle(-speed)


def stop_all_motors():
    for motor in MOTOR_PWM.values():
        motor['A1'].stop()
        motor['B1'].stop()

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)

    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
def cleanup():
    stop_all_motors()
    GPIO.cleanup()
    camera.release()
    cv2.destroyAllWindows()

vertices = np.array([[(0, 480), (100, 300), (500, 300), (640, 480)]], dtype=np.int32)

# 간단한 CNN 모델 정의
class RoadClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(RoadClassifier, self).__init__()
        self.cnn = models.shufflenet_v2_x0_5(pretrained=True)
        in_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Linear(in_features, num_classes)
    def forward(self, x):
        return self.cnn(x)

# 모델 초기화
model = RoadClassifier(num_classes=4).to(torch.device('cpu'))

# 저장된 가중치 불러오기
checkpoint = torch.load('road_classifier_model_shuffle_2.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)
model.eval()

camera = cv2.VideoCapture(0)
image = None  # 캡처된 이미지를 저장할 변수
capture_lock = threading.Lock()

# 캡처하는거랑 제어하는거랑 분리하는 함수
def capture_frames():
    global image
    while True:
        _, captured_frame = camera.read()
        captured_frame = cv2.flip(captured_frame, -1)
        # 프레임 처리 작업
        masked_frame = region_of_interest(captured_frame, vertices)

        # 락을 이용하여 이미지에 동시에 접근하지 않도록 함
        with capture_lock:
            #image = masked_frame.copy()
            image = cv2.resize(masked_frame, (224, 224))

capture_thread = threading.Thread(target=capture_frames)
capture_thread.start()

# PID 제어기 초기화
pid_controller_left = PIDController(kp=2.4, ki=0.05, kd=0.1, log=True)  # 좌측 모터 제어기 
pid_controller_right = PIDController(kp=3, ki=0.075, kd=0.1, log=True)  # 우측 모터 제어기

# 목표 상태
TARGET_STRAIGHT = [1, 0, 0, 0, 0]  # 목표 직진 상태
TARGET_LEFT = [0, 1, 0, 0, 0]      # 목표 좌회전 상태
TARGET_RIGHT = [0, 0, 1, 0, 0]     # 목표 우회전 상태
TARGET_CROSS = [0, 0, 0, 1, 0]     # 목표 횡단보도 상태

# 현재 상태 및 기본 속도 설정
base_speed_l = 28.5  # 좌측 모터의 기본 속도
base_speed_r = 50  # 우측 모터의 기본 속도

# 메인 루프 시작 전 시간 초기화
previous_time = time.time()

# 정지 카운터
cnt = 0
flag = 0

while True:
    if flag == 0:
        time.sleep(1)
        flag = 1
        
    # 현재 시간 측정
    current_time = time.time()
    dt = current_time - previous_time  # 시간 간격 계산

    previous_time = current_time  # 이전 시간 업데이트

    # 이미지 전처리
    with capture_lock:
        if image is None:
            continue  # 이미지가 캡처되지 않은 경우 스킵
        input_frame = image.copy()

    # 모델 예측 시간 측정 시작
    start_time = time.time()

    # transform 및 모델 예측
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    input_tensor = transform(input_frame)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        result = model(input_batch)
        predict = torch.argmax(result).item()
        print("\nclass:", predict)
        #print('result:', result)
        # 소프트맥스 적용
        probabilities = F.softmax(result, dim=1)
        print("probability:", probabilities)
        

    # 모델 예측 시간 측정 종료
    end_time = time.time()
    prediction_time = end_time - start_time

    print('Prediction time:', prediction_time, 'seconds')

    # 화면에 결과 표시
    cv2.putText(image, f'Prediction: {predict}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Road Classifier', image)

    # 현재 상태에 대한 목표값 설정
    if predict == 0:  # 직진
        target = TARGET_STRAIGHT
    elif predict == 1:  # 좌회전
        target = TARGET_LEFT
    elif predict == 2:  # 우회전
        target = TARGET_RIGHT
    elif predict == 3:  # 횡단보도
        target = TARGET_CROSS    

    # 현재 확률값 추출 및 오차 계산
    prob_straight = result[0][0].item()
    prob_left = result[0][1].item()
    prob_right = result[0][2].item()
    error_left_turn = prob_left - prob_right  # 좌회전과 우회전 확률의 차이     
    error_right_turn = prob_right - prob_left

    # 모터 속도 조절
    if predict == 0:  # 직진
        if prob_left > prob_right:
            right_correction = pid_controller_left.update(error_left_turn, dt)
            set_motor_speed('left', base_speed_l)
            set_motor_speed('right', base_speed_r + right_correction)
            print("left_speed : ", base_speed_l, " right_speed : ", base_speed_r + right_correction)
        elif prob_right > prob_left:
            left_correction = pid_controller_right.update(error_right_turn, dt)
            set_motor_speed('left', base_speed_l + left_correction)
            set_motor_speed('right', base_speed_r)
            print("left_speed : ", base_speed_l + left_correction, " right_speed : ", base_speed_r)
        else:
            set_motor_speed('left', base_speed_l)
            set_motor_speed('right', base_speed_r)
            print("left_speed : ", base_speed_l, " right_speed : ", base_speed_r)
        cnt = 0
    elif predict == 1:  # 좌회전
        left_turn_ml = pid_controller_left.update(error_left_turn, dt)
        left_turn_mr = pid_controller_right.update(error_left_turn, dt)
        set_motor_speed('left', base_speed_l - left_turn_ml)
        set_motor_speed('right', base_speed_r + left_turn_mr)
        print("left_speed : ", base_speed_l - left_turn_ml, " right_speed : ", base_speed_r + left_turn_mr)
        cnt = 0
    elif predict == 2:  # 우회전
        right_turn_ml = pid_controller_left.update(error_right_turn, dt)
        right_turn_mr = pid_controller_right.update(error_right_turn, dt)
        set_motor_speed('left', base_speed_l + right_turn_ml)
        set_motor_speed('right', base_speed_r - right_turn_mr)
        print("left_speed : ", base_speed_l + right_turn_ml, " right_speed : ", base_speed_r - right_turn_mr)
        cnt = 0
    elif predict == 3:  # 횡단보도
        if cnt == 0:
          set_motor_speed('left', 0) 
          set_motor_speed('right', 0) 
          cnt += 1
          time.sleep(1)
        else:
            set_motor_speed('left', base_speed_l) 
            set_motor_speed('right', base_speed_r) 

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

atexit.register(cleanup())