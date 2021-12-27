
import cv2
import os

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')    # 嘴

# To capture video from webcam.
cap = cv2.VideoCapture(0)
# To use a video file as input
# cap = cv2.VideoCapture('filename.mp4')
count = 0
while True:
    ret, frame = cap.read()  # 讀取影像
    if ret == True:

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect the faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        # Draw the rectangle around each face
        face_count = 0
        mouth_count = 0
        for (x, y, w, h) in faces:
            face_count += 1

            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            # 嘴巴偵測
            mouth = mouth_cascade.detectMultiScale(roi_gray, 1.5, 5)
            for (mx, my, mw, mh) in mouth:
                mouth_count += 1
                m = cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (0, 255, 0), 3)  # 嘴巴框繪製
                if (mouth_count == 1):
                    break

            if (face_count == 1):
                break
        cv2.imshow('Image', frame)  # 顯示影像


    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 13:  # 按下enter鍵save image
        count += 1
        fileName = str(count) + ".png"
        out_path = os.path.join('output', fileName)
        cv2.imwrite(out_path, frame)
    if k==27:
        break
# Release the VideoCapture object
cap.release()