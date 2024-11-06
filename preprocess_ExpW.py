import csv
import os
import numpy as np
import h5py
import skimage.io
import cv2 as cv

train_path = 'ExpW_align/'

train_angry_path = os.path.join(train_path, 'angry')
train_disgust_path = os.path.join(train_path, 'disgust')
train_fear_path = os.path.join(train_path, 'fear')
train_happy_path = os.path.join(train_path, 'happy')
train_sad_path = os.path.join(train_path, 'sad')
train_surprise_path = os.path.join(train_path, 'surprise')
train_neutral_path = os.path.join(train_path, 'neutral')

Data_ratio = 0.8

len_angry = 3671 * Data_ratio
len_disgust = 3995 * Data_ratio
len_fear = 1088 * Data_ratio
len_happy = 30537 * Data_ratio
len_sad = 10559 * Data_ratio
len_surprise = 7060 * Data_ratio
len_neutral = 34883 * Data_ratio

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
	I = cv.resize(I, (100, 100))	
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
	I = cv.resize(I, (100, 100))
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
	I = cv.resize(I, (100, 100))	
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
	I = cv.resize(I, (100, 100))
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
	I = cv.resize(I, (100, 100))
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
	I = cv.resize(I, (100, 100))	
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
	I = cv.resize(I, (100, 100))	
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



datafile = h5py.File('ExpW_data_100.h5', 'w')
datafile.create_dataset("train_data_pixel", dtype = 'uint8', data=train_data_x)
datafile.create_dataset("train_data_label", dtype = 'int64', data=train_data_y)
datafile.create_dataset("valid_data_pixel", dtype = 'uint8', data=valid_data_x)
datafile.create_dataset("valid_data_label", dtype = 'int64', data=valid_data_y)
datafile.close()

print("Save data finish!!!")
