#!/usr/bin/python3
# -*- coding: UTF-8 -*-

'''
import os
import shutil               #第一部分:       归类         

path = 'Emotion/'
picture_path = 'cohn-kanade-images/'

files = os.listdir(path)
for file in files:
    sub_files = os.listdir(os.path.join(path + str(file)))
    for sub_file in sub_files:
        txt_sub_files = os.listdir(os.path.join(path + str(file) + '/' + str(sub_file)))
        for txt_sub_file in txt_sub_files:
            txt = os.path.join(path + str(file) + '/' + str(sub_file) + '/' + str(txt_sub_file))
            picture3 = os.path.join(picture_path + str(file) + '/' + str(sub_file) + '/'+ str(file) \
                + '_' + str(sub_file) + '_' + str(txt_sub_file.split('_')[2]) + '.png')
            picture2 = os.path.join(picture_path + str(file) + '/' + str(sub_file) + '/'+ str(file) \
                + '_' + str(sub_file) + '_' + str(str(int(txt_sub_file.split('_')[2])-1).zfill(8)) + '.png')
            picture1 = os.path.join(picture_path + str(file) + '/' + str(sub_file) + '/'+ str(file) \
                + '_' + str(sub_file) + '_' + str(str(int(txt_sub_file.split('_')[2])-2).zfill(8)) + '.png')
            picture0 = os.path.join(picture_path + str(file) + '/' + str(sub_file) + '/'+ str(file) \
                + '_' + str(sub_file) + '_' + str(str(1).zfill(8)) + '.png')

            print (picture0)
            print (picture1)
            print (picture2)
            print (picture3)
            with open(txt,'r') as f:
                for line in f:
                    label = int(line.split('.')[0])
            print (label)
            new_picture_path = os.path.join(str(label) + '/')
            shutil.move(picture0,os.path.join(str(0) + '/'))
            shutil.move(picture1,new_picture_path)
            shutil.move(picture2,new_picture_path)
            shutil.move(picture3,new_picture_path)

'''''

















'''                                     #第二部分:       裁剪人脸         
import dlib
import face_recognition
import math
import numpy as np
import cv2
import sys
import os
from os.path import basename

 
def rect_to_bbox(rect):
    """获得人脸矩形的坐标信息"""
    # print(rect)
    x = rect[3]
    y = rect[0]
    w = rect[1] - x
    h = rect[2] - y
    return (x, y, w, h)
 
 
def face_alignment(faces):
    # 预测关键点
    #print("进行对齐-----")
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    faces_aligned = []
    for face in faces:
        rec = dlib.rectangle(0, 0, face.shape[0], face.shape[1])
        shape = predictor(np.uint8(face), rec)
        # left eye, right eye, nose, left mouth, right mouth
        order = [36, 45, 30, 48, 54]
        for j in order:
            x = shape.part(j).x
            y = shape.part(j).y
        # 计算两眼的中心坐标
        eye_center =((shape.part(36).x + shape.part(45).x) * 1./2, (shape.part(36).y + shape.part(45).y) * 1./2)
        dx = (shape.part(45).x - shape.part(36).x)
        dy = (shape.part(45).y - shape.part(36).y)
        # 计算角度
        angle = math.atan2(dy, dx) * 180. / math.pi
        # 计算仿射矩阵
        RotateMatrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)
        # 进行仿射变换，即旋转
        RotImg = cv2.warpAffine(face, RotateMatrix, (face.shape[0], face.shape[1]))
        faces_aligned.append(RotImg)
    return faces_aligned
 
 
def test(img_path):
    unknown_image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    # 定位图片中的人脸
    face_locations = face_recognition.face_locations(unknown_image)
    # 提取人脸区域的图片并保存
    src_faces = []
    src_face_num = 0
    for (i, rect) in enumerate(face_locations):
        src_face_num = src_face_num + 1
        (x, y, w, h) = rect_to_bbox(rect)
        detect_face = unknown_image[y:y+h, x:x+w]
        src_faces.append(detect_face)
        #detect_face = cv2.cvtColor(detect_face, cv2.COLOR_RGBA2BGR)
        #cv2.imwrite("result/face_" + str(src_face_num) + ".jpg", detect_face)
    # 人脸对齐操作并保存
    len_srcfaces = len(src_faces)
    print("检测到该图片的人脸数：",len_srcfaces)
    
    faces_aligned = face_alignment(src_faces)
    len_alignedfaces = len(faces_aligned) 
    print("对齐的人脸数：",len_alignedfaces)
     
    #判断：如果这张图片没有检测到人脸，将图片的路径记录到日志文本中
    if len_srcfaces == 0 and len_alignedfaces == 0:
        print("该输入图片未检测到人脸，将其写入文本中---")
        print(img_path,file=open("fail.txt","a"))
        
  
    fileName = basename(img_path)
    face_num = 0
    for faces in faces_aligned:
        face_num = face_num + 1
        #faces = cv2.cvtColor(faces, cv2.COLOR_RGBA2BGR)
        resize_image = cv2.resize(faces,(100,100))
        cv2.imwrite('croped/7/'+str(fileName), resize_image)
    pass
 
 
if __name__ == '__main__':
    #单张图片检测
    #image_file = sys.argv[1]
    #test(image_file)
    #批量图片检测，path为你的图片路径
    path = 'original/7/'
    filelist = os.listdir(path)
    #定义一个变量用于计数，你输入的图片个数
    i = 0
    for file in filelist:
        i = i+1
        img_path = path + file
        print("第{}张图片 {}".format(i,img_path))
        test(img_path)        

    #print(filelist)
    print("---------------OVER------------- !!! ")
    pass

'''

































import csv
import os
import numpy as np
import h5py
import skimage.io
import cv2 as cv

train_path = 'croped/'

train_angry_path = os.path.join(train_path, 'angry')
train_disgust_path = os.path.join(train_path, 'disgust')
train_fear_path = os.path.join(train_path, 'fear')
train_happy_path = os.path.join(train_path, 'happy')
train_sad_path = os.path.join(train_path, 'sadness')
train_surprise_path = os.path.join(train_path, 'surprise')
train_neutral_path = os.path.join(train_path, 'neutral')

Data_ratio = 0.8

len_angry = (135/3 * Data_ratio) * 3 
len_disgust = (177/3  * Data_ratio) * 3 
len_fear = (75/3  * Data_ratio) * 3 
len_happy = (207/3  * Data_ratio) * 3 
len_sad = (84/3  * Data_ratio) * 3 
len_surprise = (249/3  * Data_ratio) * 3 
len_neutral = 327 * Data_ratio

# # Creat the list to store the data and label information
train_data_x = []
train_data_y = []
valid_data_x = []
valid_data_y = []

files = os.listdir(train_angry_path)
files.sort()
for filename in files:
	print (filename)
	I = skimage.io.imread(os.path.join(train_angry_path,filename))	
	if len_angry > 0:
		len_angry = len_angry - 1
		train_data_x.append(I.tolist())
		train_data_y.append(0)
	else:
		valid_data_x.append(I.tolist())
		valid_data_y.append(0)

print (0000000000000000000)

files = os.listdir(train_disgust_path)
files.sort()
for filename in files:
	print (filename)
	I = skimage.io.imread(os.path.join(train_disgust_path,filename))
	if len_disgust > 0:
		len_disgust = len_disgust - 1
		train_data_x.append(I.tolist())
		train_data_y.append(1)
	else:
		valid_data_x.append(I.tolist())
		valid_data_y.append(1)

print (1111111111111111)

files = os.listdir(train_fear_path)
files.sort()
for filename in files:
	print (filename)
	I = skimage.io.imread(os.path.join(train_fear_path,filename))
	if len_fear > 0:
		len_fear = len_fear - 1
		train_data_x.append(I.tolist())
		train_data_y.append(2)
	else:
		valid_data_x.append(I.tolist())
		valid_data_y.append(2)

print (22222222222222222222)

files = os.listdir(train_happy_path)
files.sort()
for filename in files:
	print (filename)
	I = skimage.io.imread(os.path.join(train_happy_path,filename))
	if len_happy > 0:
		len_happy = len_happy - 1
		train_data_x.append(I.tolist())
		train_data_y.append(3)
	else:
		valid_data_x.append(I.tolist())
		valid_data_y.append(3)

print (333333333333333333)

files = os.listdir(train_sad_path)
files.sort()
for filename in files:
	print (filename)
	I = skimage.io.imread(os.path.join(train_sad_path,filename))
	if len_sad > 0:
		len_sad = len_sad - 1
		train_data_x.append(I.tolist())
		train_data_y.append(4)
	else:
		valid_data_x.append(I.tolist())
		valid_data_y.append(4)

print (444444444444444444444444)

files = os.listdir(train_surprise_path)
files.sort()
for filename in files:
	print (filename)
	I = skimage.io.imread(os.path.join(train_surprise_path,filename))
	if len_surprise > 0:
		len_surprise = len_surprise - 1
		train_data_x.append(I.tolist())
		train_data_y.append(5)
	else:
		valid_data_x.append(I.tolist())
		valid_data_y.append(5)

print (555555555555555555555555)

files = os.listdir(train_neutral_path)
files.sort()
for filename in files:
	print (filename)
	I = skimage.io.imread(os.path.join(train_neutral_path,filename))	
	if len_neutral > 0:
		len_neutral = len_neutral - 1
		train_data_x.append(I.tolist())
		train_data_y.append(6)
	else:
		valid_data_x.append(I.tolist())
		valid_data_y.append(6)

print (666666666666666666666666)


print(np.shape(train_data_x))
print(np.shape(train_data_y))

print (999999999999999999999999)

print(np.shape(valid_data_x))
print(np.shape(valid_data_y))



datafile = h5py.File('CK_Plus_data_100.h5', 'w')
datafile.create_dataset("train_data_pixel", dtype = 'uint8', data=train_data_x)
datafile.create_dataset("train_data_label", dtype = 'int64', data=train_data_y)
datafile.create_dataset("valid_data_pixel", dtype = 'uint8', data=valid_data_x)
datafile.create_dataset("valid_data_label", dtype = 'int64', data=valid_data_y)
datafile.close()

print("Save data finish!!!")













