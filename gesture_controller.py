import cv2
import time
import math

import utils.hand_tracking_module as htm

# 高さと横幅の設定
wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

p_time = 0

detector = htm.HandDetector(detection_con=0.7)

while True:
    success, img = cap.read()
    img = detector.find_hands(img)
    lm_list = detector.find_position(img, draw=False)

    if len(lm_list) != 0:
        # 親指の位置を取得
        x1, y1 = lm_list[4][1], lm_list[4][2]
        # 人差し指の位置を取得
        x2, y2 = lm_list[8][1], lm_list[8][2]
        # 親指と人差し指の中心の座標を取得
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # それぞれの部位の円を大きくする
        cv2.circle(img, (x1, y1), 15, (0, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (0, 0, 255), cv2.FILLED)
        cv2.circle(img, (cx, cy), 15, (0, 0, 255), cv2.FILLED)

        # 親指と人差し指に直線を引く
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        length = math.hypot(x2 - x1, y2 - y1)

        if length <= 50:
            # 指を閉じたら緑色になる
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time

    cv2.putText(img, f'FPS: {int(fps)}', (40, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (0, 255, 0), 3)
    cv2.imshow('Img', img)

    # qキーを押すことで停止する
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
