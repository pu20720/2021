# encoding=utf-8
# OpenCV 4.0: 人臉位置偵測
# pip install opencv-python==4.4.0.46

import cv2
import os
from numpy import nan, true_divide
from sklearn.metrics import confusion_matrix
from collections import namedtuple

Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

def Iou(a, b):  
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if(dx<0):
        dx=0
    if(dy<0):
        dy=0
    
    a_w=a.xmax-a.xmin # a長度(實際)
    a_h=a.ymax-a.ymin # a高度
    b_w=b.xmax-b.xmin # b長度(測試)
    b_h=b.ymax-b.ymin # b高度
    intersection = dx * dy  # 交集
    union = a_w*a_h + b_w*b_h - intersection  # 聯集
    print("交集:", intersection)
    print("聯集:", union)
    iou=intersection / union


    return iou

    
        
def area(a):  
    if(a.xmax == 0 & a.xmin== 0 & a.ymax== 0 & a.ymin== 0):
        return 0

     

# Step 1 人臉級聯分類器(CascadeClassifier)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')    # 臉
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')    # 嘴


actual = []

predicted = []
predicted_rects = []
actual_rects = []
                
count = 0
dirs = os.listdir("input")
for d in dirs:
    files = os.listdir("input/{d}".format(d=d))
    for i in files:   
    
        img_path = os.path.join('input', d, i)

        # Step 2  讀取輸入影像  
        img = cv2.imread(img_path)                 

        #img = cv2.imread('./input/sr10.jpg')

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Step 3  人臉偵測
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Step 4  人臉繪製與儲存
        face_count = 0
        mouth_count = 0
        for (x, y, w, h) in faces:
            face_count += 1
            img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 5)  # 人臉框繪製
            f = cv2.resize(img[y:y+h, x:x+w], (224, 224))           # 224x224 CNN 輸入大小

 
            if(face_count == 1):
                box = Rectangle(x, y, x+w, y+h)
                predicted_rects.append(box)
                actual.append(1)
                break

        if(face_count == 0):
            box = Rectangle(0,0,0,0)
            predicted_rects.append(box)
            actual.append(0)
                


        # 實際人臉框繪製
        img = cv2.rectangle(img, (actual_rects[count].xmin, actual_rects[count].ymin), (actual_rects[count].xmax, actual_rects[count].ymax), (34,139,34), 5) 
        # save img
        path = os.path.join('output', d)
        if not os.path.exists(path):
            os.mkdir(path)                  
        out_path = os.path.join('output', d, i)
        cv2.imwrite(out_path, img)


        count += 1
        print(i)


iou_arr = []
for i in range(0, len(actual)):      #判斷圈出的地方
    if(area(actual_rects[i]) != 0):  # 照片中真的有臉的
        if(area(predicted_rects[i]) !=0):   # 程式有框出東西(不論是框對框錯)
            iou = Iou(actual_rects[i], predicted_rects[i])    # 計算IOU
            if(iou > 0.7):   # 框對 TP
                predicted.append(1)
                print(actual_rects[i])
                #print(predicted_rects[i])
        
            else:  # 框錯  FN
                predicted.append(0)
                actual[i] = 1
                print(actual_rects[i])
               # print("框錯")
        else:   # 沒框到  FN
            predicted.append(0)
            actual[i] = 1
            print(actual_rects[i])
            #print("沒框到")
            iou = 0
        iou_arr.append(iou)
        print("IOU:", iou)
    else:   # 照片中沒有臉的 
        if(area(predicted_rects[i]) != 0):    # 有框到東西 FP
            predicted.append(1) 
            actual[i] = 0 
            iou_arr.append(nan)
            print("IOU:", nan)
        else:               # TN
            predicted.append(0)            
            iou_arr.append("Na")
            print("IOU:Na")  
        print(actual_rects[i])
 
  
print("實際:",actual)
print("預測:",predicted)
tn, fp, fn, tp = confusion_matrix(actual, predicted).ravel()  # 判斷陰性陽性
print("tn, fp, fn, tp:",tn, fp, fn, tp)
print("準確率(Precision):", tp / (tp + fp))
print("回復率(Recall):", tp/(tp+fn))
iou_arr.append(tp / (tp + fp))
iou_arr.append(tp/(tp+fn))
print(iou_arr)
