import cv2
import mediapipe as mp
import numpy as np
import math # noi suy diem cho muot
from utils import draw_interpolated_line

mp_hands = mp.solutions.hands            # module Hands của MediaPipe
mp_draw = mp.solutions.drawing_utils     # công cụ vẽ landmark/connection

# object to track hand
hands = mp_hands.Hands(
    max_num_hands=1,# only one hand                    
    min_detection_confidence=0.7,      
    min_tracking_confidence=0.7        
)


cap = cv2.VideoCapture(0) # cam laptop: 0           
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
canvas = None
prev_x, prev_y = None, None

while True:
    ret, frame = cap.read()
    if not ret:
        break                    

    # flip like mirror
    frame = cv2.flip(frame, 1)

    # frame's shape
    h, w, c = frame.shape

    if canvas is None:
        canvas = np.zeros_like(frame)  # create canvas

    # MediaPipe yêu cầu ảnh RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)       

    current_color= (255, 255, 255)
    #
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            ngontro_up = handLms.landmark[8].y < handLms.landmark[6].y
            ngongiua_up = handLms.landmark[12].y < handLms.landmark[10].y

            lm = handLms.landmark[8]
            x, y = int(lm.x *w), int(lm.y *h)
            if ngontro_up and not ngongiua_up:
                current_color = (255,255,255)

            if ngontro_up and ngongiua_up:
                current_color = (0,0,255)

            if prev_x is not None and prev_y is not None and ngontro_up:
                prev = (prev_x, prev_y)
                cur = (x,y)
                # prev (func utils) polylines -> tra ve x,y
                prev = draw_interpolated_line(canvas, prev, cur, current_color, 5, 5, 50)
                prev_x, prev_y = prev
            else: 
                if ngontro_up: # ngon tro + ko co prev: x,y la diem bat dau
                    prev_x, prev_y = x, y
                else: #khong thay ngon tro: ko ve
                    prev_x, prev_y = None, None

            # Vẽ landmarks và connections lên frame để debug/quan sát
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
    else:
        # Nếu không thấy bàn tay thì reset prev để tránh nối đường nhảy lung tung
        prev_x, prev_y = None, None

    # Trộn canvas (nét vẽ) lên frame để hiển thị cùng lúc
    # addWeighted(src1, alpha, src2, beta, gamma)
    merged = cv2.addWeighted(frame, 0.5, canvas, 1, 0)

    # show windows, put canvas + line on it
    cv2.imshow("Air Drawing", merged)

    key = cv2.waitKey(1)
    if key == 27:
        cv2.imwrite("canvas.png", canvas)
        break

cap.release()
cv2.destroyAllWindows()
