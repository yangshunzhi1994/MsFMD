#!/usr/bin/python3
# -*- coding: UTF-8 -*-
import cv2
import numpy as np
import torch
import random
from torch.autograd import Variable
from studentNet import CNN_RIS

def sp_noise(image, prob):
    ''' 添加椒盐噪声 '''
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

def preprocess_image(cv2im, mean, std, resize_Teacher=True):
    ''' 图像预处理 '''
    if resize_Teacher:
        cv2im = cv2.resize(cv2im, (92, 92))
    else:
        cv2im = cv2.resize(cv2im, (44, 44))
    im_as_arr = np.float32(cv2im)
    im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])
    im_as_arr = im_as_arr.transpose(2, 0, 1)
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    im_as_ten = torch.from_numpy(im_as_arr).float()
    im_as_ten.unsqueeze_(0)
    im_as_var = Variable(im_as_ten, requires_grad=False)
    return im_as_var

# 模型加载
snet = CNN_RIS()
scheckpoint = torch.load('results/Student_Test_model.t7', map_location='cpu')
snet.load_state_dict(scheckpoint['snet'], strict=False)
snet.eval()

# 均值和标准差
tmean = [0.5884594, 0.45767313, 0.40865755]
tstd = [0.25717735, 0.23602168, 0.23505741]

# 原始图像
img = cv2.imread('image/707.jpg')
img = cv2.resize(img, (92, 92))  # resize 为一致大小，便于观测效果

# 所有噪声类型和处理
noises = {
    'Clean': img,
    'GaussianBlur': cv2.GaussianBlur(img, (5, 5), 0),
    'AverageBlur': cv2.blur(img, (5, 5)),
    'MedianBlur': cv2.medianBlur(img, 5),
    'BilateralFilter': cv2.bilateralFilter(img, 10, 100, 100),
    'Salt-and-pepper': sp_noise(img, prob=0.05)
}

# 对每种噪声处理的图像做预测
for noise_type, noise_img in noises.items():
    simg = preprocess_image(noise_img, tmean, tstd, resize_Teacher=False)
    with torch.no_grad():
        _, _, _, _, out_s = snet(simg)
        pred = torch.argmax(out_s, dim=1).item()
        print(f'{noise_type} → Prediction: {pred} | Raw Output: {out_s}')