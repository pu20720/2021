import cv2
import os

# SQ13 IP Camera 的擷取網址
URL = "http://192.168.25.1:8080/?action=stream"
ipcam = cv2.VideoCapture(URL)  # 開啟 IP Camera
ipcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 設定影像寬度  (SQ13 最大 1920)
ipcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 設定影像高度  (SQ13 最大高度 1080)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')    # 嘴

count = 0
while True:  # 無窮迴圈擷取影像
    ret, frame = ipcam.read()  # 讀取影像
    if ret == True:

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect the faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        # Draw the rectangle around each face
        for (x, y, w, h) in faces:
            frame=cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            # 嘴巴偵測
            mouth = mouth_cascade.detectMultiScale(roi_gray, 1.5, 5)
            for (mx, my, mw, mh) in mouth:
                m = cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (0, 255, 0), 3)  # 嘴巴框繪製
            continue



        cv2.imshow('Image', frame)  # 顯示影像

    if cv2.waitKey(1) == 13:  # 按下enter鍵save image
        count += 1
        fileName = str(count) + ".png"
        out_path = os.path.join('output', fileName)
        cv2.imwrite(out_path, frame)

    if cv2.waitKey(1) == 27:  # 按下Esc鍵結束
        ipcam.release()
        cv2.destroyAllWindows()

        break