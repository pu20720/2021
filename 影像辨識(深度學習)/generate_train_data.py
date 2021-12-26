import os
import cv2
import dlib
import time
import argparse
import numpy as np
from imutils import video

DOWNSAMPLE_RATIO = 4

'''
-----------------
| (x1,y1)       |
|       (x2,y2) |
----------------*
                原點
'''
def findBound(mask):
    x1,x2,y1,y2 = 0,0,0,0
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    for i in range(gray.shape[0]):
        if np.where(gray[i,:]==0)[0].size > 1:
            x2 = i
            break
    for i in range (gray.shape[0]-1,0,-1):
        if np.where(gray[i,:]==0)[0].size > 1:
            x1 = i
            break
    for i in range(gray.shape[1]):
        if np.where(gray[:,i]==0)[0].size > 1:
            y2 = i
            break
    for i in range (gray.shape[1]-1,0,-1):
        if np.where(gray[:,i]==0)[0].size >1:
            y1 = i
            break
    return x1,y1,x2,y2


def regularization(mask):
    white = 255
    halfSize = mask.shape[0]//2
    resize = (halfSize,halfSize)
    quarterSize = halfSize//2
    white_backGroup = np.full(mask.shape, white, np.uint8)
    x1,y1,x2,y2 = findBound(mask)
    #print((x1,y1),(x2,y2))
    if x1!=0 and x2!=0 and y1!=0 and y2!=0:
        mask_resize = cv2.resize(mask[x2:x1, y2:y1], resize , cv2.INTER_AREA)
        #將mask_resize 放入白背景正中間
        white_backGroup[halfSize-quarterSize : halfSize+quarterSize, halfSize-quarterSize : halfSize+quarterSize] = mask_resize
        return white_backGroup, mask_resize
    return white_backGroup, mask

def reshape_for_polyline(array):
    return np.array(array, np.int32).reshape((-1, 1, 2))


def main():
    os.makedirs('train', exist_ok=True)
    os.makedirs('train/trainB', exist_ok=True)
    os.makedirs('train/trainA', exist_ok=True)

    cap = cv2.VideoCapture(args.filename)
    fps = video.FPS().start()

    count = 0
    while cap.isOpened():
        ret, frame = cap.read()

        frame_resize = cv2.resize(frame, (256,256), cv2.INTER_AREA)
        gray = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1)
        white_img = np.full(frame_resize.shape, 255, np.uint8)
        t = time.time()

        # Perform if there is a face detected
        if len(faces) == 1:
            for face in faces:
                detected_landmarks = predictor(gray, face).parts()
                landmarks = [[p.x, p.y] for p in detected_landmarks]

                jaw = reshape_for_polyline(landmarks[0:17])
                left_eyebrow = reshape_for_polyline(landmarks[22:27])
                right_eyebrow = reshape_for_polyline(landmarks[17:22])
                nose_bridge = reshape_for_polyline(landmarks[27:31])
                lower_nose = reshape_for_polyline(landmarks[30:35])
                left_eye = reshape_for_polyline(landmarks[42:48])
                right_eye = reshape_for_polyline(landmarks[36:42])
                outer_lip = reshape_for_polyline(landmarks[48:60])
                inner_lip = reshape_for_polyline(landmarks[60:68])

                color = (0, 0, 0)
                thickness = 3

                cv2.polylines(white_img, [jaw], False, color, thickness)
                cv2.polylines(white_img, [left_eyebrow], False, color, thickness)
                cv2.polylines(white_img, [right_eyebrow], False, color, thickness)
                cv2.polylines(white_img, [nose_bridge], False, color, thickness)
                cv2.polylines(white_img, [lower_nose], True, color, thickness)
                cv2.polylines(white_img, [left_eye], True, color, thickness)
                cv2.polylines(white_img, [right_eye], True, color, thickness)
                cv2.polylines(white_img, [outer_lip], True, color, thickness)
                cv2.polylines(white_img, [inner_lip], True, color, thickness)

            # Display the resulting frame
            count += 1
            print(count)
            #print('frame_resize',white_img.shape)
            cv2.imwrite("train/trainB/{}.png".format(count), frame_resize)
            regularizationImg, maskHelf = regularization(white_img)
            cv2.imwrite("train/trainA/{}.png".format(count), regularizationImg)
            fps.update()
            combine = np.concatenate([regularizationImg, frame_resize], axis=1)  
            #cv2.imshow('combine',combine)
            print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))

            if count == args.number:  # only take 400 photos
                break
            elif cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("No face detected")

    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', dest='filename',default='ho.mp4', type=str, help='Name of the video file.')
    parser.add_argument('--num', dest='number',default=800, type=int, help='Number of train data to be created.')
    parser.add_argument('--landmark-model', dest='face_landmark_shape_file', default='shape_predictor_68_face_landmarks.dat' ,type=str, help='Face landmark model file.')
    args = parser.parse_args()

    # Create the face predictor and landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.face_landmark_shape_file)

    main()