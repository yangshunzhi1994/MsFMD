import cv2
import numpy as np
import random
from PIL import Image

def sp_noise(image, prob):
    '''
    添加椒盐噪声
    prob:噪声比例 
    '''
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

# 读取图片（修改为你的图片路径）
img = Image.open('image/707.jpg')  # 例如 'input.jpg'
img_rgb = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

# 高斯模糊
gaussian = cv2.GaussianBlur(img_rgb, (5, 5), 0)
cv2.imwrite('gaussian_blur.jpg', gaussian)

# 均值模糊
average = cv2.blur(img_rgb, (5, 5))
cv2.imwrite('average_blur.jpg', average)

# 中值模糊
median = cv2.medianBlur(img_rgb, 5)
cv2.imwrite('median_blur.jpg', median)

# 双边滤波
bilateral = cv2.bilateralFilter(img_rgb, 10, 100, 100)
cv2.imwrite('bilateral_filter.jpg', bilateral)

# 椒盐噪声
sp = sp_noise(img_rgb, prob=0.05)
cv2.imwrite('salt_pepper.jpg', sp)

print("所有带噪图像已保存！")