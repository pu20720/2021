# -*- coding: utf-8 -*-
import cv2
import numpy as np
import dlib
from keras.models import load_model
import tensorflow as tf
from keras_contrib.layers import InstanceNormalization
#tf.compat.v1.disable_eager_execution()
import argparse

# for regularImg find face bound
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


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
        ret, frame = cap.read()
        if args.model_imgSize == 128:
            cv2.imshow('fame',AtoB128(frame))
        else:
            cv2.imshow('fame',AtoB256Regular(frame))
            #cv2.imshow('fame',AtoB256(frame))
        cho = cv2.waitKey(30) & 0xFF
        if cho == ord('q'):
            break       
    cap.release()
    cv2.destroyAllWindows()
    
    
def AtoB128(frame):
    mask = img2polyline(frame)
    mask_resize = cv2.resize(mask,(128, 128), cv2.INTER_AREA)
    mask_resize = np.array(mask_resize)/127.5 - 1
    img_expamd_dim = np.expand_dims(mask_resize, axis=0)
    img_predict = ganerator.predict(img_expamd_dim)
    #print(img_expamd_dim.shape)
    image_bgr = cv2.cvtColor(np.squeeze(img_predict), cv2.COLOR_RGB2BGR)
    image_bgr = np.array(image_bgr)*0.5 +0.5
    imgResize = cv2.resize(image_bgr, (256, 256), cv2.INTER_AREA)
    imgCombine = np.concatenate([mask, imgResize],axis = 1)
    return imgCombine

def BtoA128(frame):  
    img256 = cv2.resize(frame,(256,256),cv2.INTER_AREA)
    img256 = np.array(img256.astype(float))/127.5 - 1
    
    frame = cv2.cvtColor(np.squeeze(frame), cv2.COLOR_RGB2BGR)
    img = cv2.resize(frame,(128, 128), cv2.INTER_AREA)
    
    img_expamd_dim = np.expand_dims(img, axis=0)
    img_expamd_dim = np.array(img_expamd_dim.astype(float))/127.5 - 1
    img_predict = ganerator.predict(img_expamd_dim)
    #print(img_expamd_dim.shape)
    image = np.array(img_predict)*0.5 +0.5
    imgResize = cv2.resize(image[0], (256, 256), cv2.INTER_AREA)
    imgCombine = np.concatenate([img256, imgResize],axis = 1)
    return imgCombine

def AtoB256(frame):
    frame_resize = cv2.resize(frame,(256,256),cv2.INTER_AREA)
    mask = img2polyline(frame_resize)
    mask = np.array(mask)/127.5 - 1
    img_expamd_dim = np.expand_dims(mask, axis=0)
    img_predict = ganerator.predict(img_expamd_dim)
    image_bgr = cv2.cvtColor(np.squeeze(img_predict), cv2.COLOR_RGB2BGR)
    image_bgr = np.array(image_bgr)*0.5 +0.5
    imgCombine = np.concatenate([mask,image_bgr],axis=1)
    return imgCombine
### To-do AtoB256 finish
def AtoB256Regular(frame):
    frame_resize = cv2.resize(frame,(256,256),cv2.INTER_AREA)
    mask = img2polyline(frame_resize)
    regularImg , mask_resize = regularization(mask)
    mask_resize = np.array(regularImg)/127.5 - 1
    img_expamd_dim = np.expand_dims(mask_resize, axis=0)
    img_predict = ganerator.predict(img_expamd_dim)
    #print(img_expamd_dim.shape)
    image_bgr = cv2.cvtColor(np.squeeze(img_predict), cv2.COLOR_RGB2BGR)
    image_bgr = np.array(image_bgr)*0.5 +0.5
    imgCombine = np.concatenate([mask, regularImg, image_bgr],axis = 1)
    return imgCombine

### To-do BtoA256
def BtoA256(frame):
    pass
  
def reshape_for_polyline(array):
    return np.array(array, np.int32).reshape((-1, 1, 2))

def img2polyline(frame):
    frame = cv2.resize(frame,(256, 256), cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)      
    white_img = np.full(frame.shape, 255, np.uint8)
    #print(frame.shape)
    for face in faces:
            
        detected_landmarks = predictor(gray, face).parts()
        landmarks = [[p.x , p.y] for p in detected_landmarks]

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
    return white_img
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    parser.add_argument('--size', dest='model_imgSize', type=int, default=256, choices=[128, 256],help='model_inputImgSize')
    parser.add_argument('--mode', dest='mode', type=int, default=1, help='0 is toBlackWhiteFace ,1 is toNormalFace')
    parser.add_argument('--model', dest='model', type=str, default='model200epochs_v4.h5', help='Frozen TensorFlow model file.')
    args = parser.parse_args()
    ganerator = load_model(args.model, custom_objects={'InstanceNormalization':InstanceNormalization})
    main()