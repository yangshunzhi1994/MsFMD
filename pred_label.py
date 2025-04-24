#!/usr/bin/python3
# -*- coding: UTF-8 -*-
import cv2
import numpy as np
import torch
import random
import os
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

# 遍历图片
for idx in range(1, 1001):
    img_path = f'image/{idx}.jpg'
    if not os.path.exists(img_path):
        print(f'文件不存在: {img_path}')
        continue

    img = cv2.imread(img_path)
    if img is None:
        print(f'图像读取失败: {img_path}')
        continue

    img = cv2.resize(img, (92, 92))  # 保持一致大小
    clean_img = img.copy()
    sp_img = sp_noise(img, prob=0.05)

    # 预处理
    input_clean = preprocess_image(clean_img, tmean, tstd, resize_Teacher=False)
    input_sp = preprocess_image(sp_img, tmean, tstd, resize_Teacher=False)

    with torch.no_grad():
        _, _, _, _, out_clean = snet(input_clean)
        _, _, _, _, out_sp = snet(input_sp)
        pred_clean = torch.argmax(out_clean, dim=1).item()
        pred_sp = torch.argmax(out_sp, dim=1).item()

        if pred_clean != pred_sp:
            print(f'找到样本：{img_path} → Clean: {pred_clean}, Salt-and-pepper: {pred_sp}')
            break