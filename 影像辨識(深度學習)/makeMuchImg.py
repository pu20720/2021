# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 16:12:18 2021

@author: a2482
"""
import cv2
from math import *
import numpy as np
import random
import copy
from glob import glob

''' 旋轉angle角度，缺失背景白色（255, 255, 255) '''
def rotate_bound_white_bg(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    # -angle位置引數為角度引數負值表示順時針旋轉; 1.0位置引數scale是調整尺寸比例（影象縮放引數），建議0.75
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    # borderValue 缺失背景填充色彩，此處為白色，可自定義
    return cv2.warpAffine(image, M, (nW, nH),borderValue=(255,255,255))
    # borderValue 預設，預設是黑色（0, 0 , 0）
    # return cv2.warpAffine(image, M, (nW, nH))
    
def gasuss_noise(image, mean=0, var=0.001):
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var**0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1
    else:
        low_clip = 0
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    noise = noise*255
    return out

'''胡椒鹽'''
def sp_noise(image, prob):
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 255
            
            elif rdn > thres:
                output[i][j] = 0
            else:
                output[i][j] = image[i][j]
    return output

# To - Do 遮蔽圖

def mask_block(img):
    '''
    -----------------
    | (x1,y1)       |
    |       (x2,y2) |
    ----------------*
                    原點
    '''
    halfSize = img.shape[0]//2
    quarterSize = halfSize//2
    center = (halfSize-quarterSize, halfSize+quarterSize)    # 64, 192 
    mask_bounding = np.random.randint(low=center[0], high=center[1], size=2)
    #print(mask_bounding)
    maskLength = 30
    imgMask_block = copy.deepcopy(img)
    imgMask_block[mask_bounding[0]-maskLength:mask_bounding[0]+maskLength, 
                  mask_bounding[1]-maskLength:mask_bounding[1]+maskLength:, ] = 255
    return imgMask_block
    


def saveInPutImage(imgPath, trainInputPath):
    filename = imgPath.split('\\')[1].split('.')[0]
    img = cv2.imread(imgPath)
    '''旋轉'''
    #正旋轉
    radius = 2
    imgTRotation = rotate_bound_white_bg(img, radius)
    imgTRotation = cv2.resize(imgTRotation,img.shape[:2], cv2.INTER_AREA)
    #負旋轉
    imgRotation = rotate_bound_white_bg(img, -radius)
    imgRotation = cv2.resize(imgRotation,img.shape[:2], cv2.INTER_AREA)
    
    '''高斯噪聲'''
    imgGasuss_noise = gasuss_noise(img)
    
    '''胡椒鹽噪聲'''
    imgSp_noise = sp_noise(img, 0.4)
    
    '''遮蔽'''
    maskBlock= mask_block(img)
    
    '''膨脹'''
    dilation = cv2.dilate(img, np.ones((3,3)))
    
    '''侵蝕'''
    erosion = cv2.erode(img, np.ones((3,3)))
    cv2.imwrite(trainInputPath+'{}_imgTRotation.png'.format(filename),imgTRotation)
    cv2.imwrite(trainInputPath+'{}_imgRotation.png'.format(filename),imgRotation)   
    cv2.imwrite(trainInputPath+'{}_imgGasuss_noise.png'.format(filename),imgGasuss_noise)
    cv2.imwrite(trainInputPath+'{}_imgSp_noise.png'.format(filename),imgSp_noise)
    cv2.imwrite(trainInputPath+'{}_maskBlock.png'.format(filename),maskBlock)
    cv2.imwrite(trainInputPath+'{}_dilation.png'.format(filename),dilation)
    cv2.imwrite(trainInputPath+'{}_erosion.png'.format(filename),erosion)
    print('{} is ok'.format(filename))

def generatortrainInputImg(trainInputPaths, trainInputLoadPath):
    for imgPath in trainInputPaths:
       saveInPutImage(imgPath, trainInputLoadPath)

def generatorTrainOutputImg(trainOutputPaths, trainOutPutLoadPath):
    for imgPath in trainOutputPaths:
        filename = imgPath.split('\\')[1].split('.')[0]
        img = cv2.imread(imgPath)
        
        cv2.imwrite(trainOutPutLoadPath+'{}_imgTRotation.png'.format(filename),img)
        cv2.imwrite(trainOutPutLoadPath+'{}_imgRotation.png'.format(filename),img)   
        cv2.imwrite(trainOutPutLoadPath+'{}_imgGasuss_noise.png'.format(filename),img)
        cv2.imwrite(trainOutPutLoadPath+'{}_imgSp_noise.png'.format(filename),img)
        cv2.imwrite(trainOutPutLoadPath+'{}_maskBlock.png'.format(filename),img)
        cv2.imwrite(trainOutPutLoadPath+'{}_dilation.png'.format(filename),img)
        cv2.imwrite(trainOutPutLoadPath+'{}_erosion.png'.format(filename),img)
        
        print('{} is ok'.format(filename))

if __name__ == '__main__':
    data_dir = 'train'
    trainInputPaths = glob(data_dir+r'/trainA/*')
    trainOutputPaths = glob(data_dir+r'/trainB/*')
    trainInputLoadPath = data_dir+r'/trainA/'
    trainOutPutLoadPath = data_dir+r'/trainB/'
    
    generatortrainInputImg(trainInputPaths, trainInputLoadPath)   #建立干擾圖片inputData
    
    generatorTrainOutputImg(trainOutputPaths, trainOutPutLoadPath) #建立干擾圖片outputData
'''
cv2.imshow("img",img)
cv2.imshow("imgRotation",imgRotation)
cv2.imshow("imgTRotation",imgTRotation)
cv2.imshow('maskBlock',maskBlock)
cv2.imshow('imgGass_noise', imgGasuss_noise)
cv2.imshow('imgSp_noise',imgSp_noise)
cv2.imshow('dilation', dilation)
cv2.imshow('erosion', erosion)

cv2.waitKey(0)
cv2.destroyAllWindows()
'''