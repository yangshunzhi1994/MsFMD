from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F





def adjust_lr(optimizer, epoch, args_lr):
	scale   = 0.1
	lr_list =  [args_lr] * 100
	lr_list += [args_lr*scale] * 50
	lr_list += [args_lr*scale*scale] * 50

	lr = lr_list[epoch-1]
	print ('Epoch: {}  lr: {:.3f}'.format(epoch, lr))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.conv0  = nn.Conv2d(96, 96, kernel_size=3, padding=1, bias=True)
        self.ReLU0 = nn.ReLU(inplace=True)
        self.Dconv0  = torch.nn.PixelShuffle(2)
        self.DReLU0 = nn.ReLU(inplace=True)
        self.conv1  = nn.Conv2d(24, 24, kernel_size=3, padding=1, bias=True)
        self.ReLU1 = nn.ReLU(inplace=True)

        self.conv2  = nn.Conv2d(24, 24, kernel_size=3, padding=1, bias=True)
        self.ReLU2 = nn.ReLU(inplace=True)
        self.Dconv1 = torch.nn.PixelShuffle(2)
        self.DReLU1 = nn.ReLU(inplace=True)
        self.conv3  = nn.Conv2d(6, 6, kernel_size=3, padding=1, bias=True)
        self.ReLU3 = nn.ReLU(inplace=True)

        self.conv4  = nn.Conv2d(6, 6, kernel_size=3, padding=1, bias=True)
        self.ReLU4 = nn.ReLU(inplace=True)
        self.conv5  = nn.Conv2d(6, 6, kernel_size=3, padding=1, bias=True)
        self.ReLU5 = nn.ReLU(inplace=True)
        self.conv6  = nn.Conv2d(6, 3, kernel_size=3, padding=1, bias=True)
        self.ReLU6 = nn.ReLU(inplace=True)


    def forward(self, x):

        x = self.ReLU0(self.conv0(x))
        x0 = self.DReLU0(self.Dconv0(x))
        x1 = self.ReLU1(self.conv1(x0)) + x0

        x = self.ReLU2(self.conv2(x1))
        x2 = self.DReLU1(self.Dconv1(x))
        x3 = self.ReLU3(self.conv3(x2)) + x2

        x = self.ReLU4(self.conv4(x3))
        x = torch.nn.functional.interpolate(x, size=[92, 92], mode='nearest', align_corners=None)

        x = self.ReLU5(self.conv5(x)) + x
        x = self.ReLU6(self.conv6(x))
        
        return x

def Absdiff_Similarity(teacher, student):
	B, C, teacher_H, teacher_W = teacher.shape
	B, C, student_H, student_W = student.shape

	teacher_norm = teacher.norm(p=2, dim=2)
	student_norm = student.norm(p=2, dim=2)
	#teacher_norm = torch.div(teacher_norm, teacher_H ** 0.5)
	#student_norm = torch.div(student_norm, student_H ** 0.5)
	teacher_norm = torch.mean(teacher_norm,dim=2,keepdim=False)
	student_norm = torch.mean(student_norm,dim=2,keepdim=False)
	absdiff = torch.abs(teacher_norm - student_norm)

	teacher = torch.nn.functional.interpolate(teacher, size=[student_H, student_W], mode='nearest', align_corners=None)
	cosineSimilarity = torch.nn.CosineSimilarity(dim=2, eps=1e-6)(teacher, student)
	cosineSimilarity = 1 - cosineSimilarity
	cosineSimilarity = torch.mean(cosineSimilarity,dim=2,keepdim=False)

	total = absdiff + cosineSimilarity
	C = 0.6*torch.max(total).item()
	loss = torch.mean(torch.where(total < C, total, (total*total+C*C)/(2*C)))
	
	return loss

class KL_divergence(nn.Module):
    def __init__(self, temperature = 1):
        super(KL_divergence, self).__init__()
        self.T = temperature
    def forward(self, teacher_logit, student_logit):

        KD_loss = nn.KLDivLoss()(F.log_softmax(student_logit/self.T,dim=1), F.softmax(teacher_logit/self.T,dim=1)) * self.T * self.T

        return KD_loss


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(c * d)

def styleLoss(teacher_input, student_feature, MSE_crit):
    teacher_input = gram_matrix(teacher_input.cuda())
    student_feature = gram_matrix(student_feature.cuda())
    loss = MSE_crit(teacher_input, student_feature)
    return loss


def confusion_matrix(preds, y, NUM_CLASSES=7):
    """ Returns confusion matrix """
    assert preds.shape[0] == y.shape[0], "1 dim of predictions and labels must be equal"
    rounded_preds = torch.argmax(preds,1)
    conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES))
    for i in range(rounded_preds.shape[0]):
        predicted_class = rounded_preds[i]
        correct_class = y[i]
        conf_mat[correct_class][predicted_class] += 1
    return conf_mat











# 定义通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y))
        return x * y

# 定义带有注意力机制的残差块
class ResidualBlockWithAttention(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlockWithAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.ca = ChannelAttention(in_channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.ca(out)
        out += residual
        out = self.relu(out)
        return out




# 定义新的 Decoder1 模块
class Decoder1(nn.Module):
    def __init__(self):
        super(Decoder1, self).__init__()

        self.resblock1 = ResidualBlockWithAttention(96)

        # 上采样到 [B, 96, 46, 46]
        self.up1 = nn.Upsample(size=(46, 46), mode='bilinear', align_corners=True)

        self.resblock2 = ResidualBlockWithAttention(96)

        # 上采样到 [B, 96, 92, 92]
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.resblock3 = ResidualBlockWithAttention(96)

        # 生成最终输出 x3
        self.conv2 = nn.Conv2d(96, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(3)

    def forward(self, x):
        x = self.resblock1(x)
        x = self.up1(x)  # 上采样到 [B, 96, 46, 46]
        x = self.resblock2(x)
        x1 = x  # 输出 x1，用于与 rb1_t 进行特征蒸馏

        x = self.up2(x1)  # 上采样到 [B, 96, 92, 92]
        x = self.resblock3(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x3 = self.relu(self.bn3(self.conv3(x)))  # 输出 x3，用于与 img_teacher 进行特征蒸馏

        return x1, x3






class Decoder2(nn.Module):
    def __init__(self):
        super(Decoder2, self).__init__()

        self.resblock1 = ResidualBlockWithAttention(160)

        # 上采样到 [B, 160, 23, 23]
        self.up1 = nn.Upsample(size=(23, 23), mode='bilinear', align_corners=True)

        self.resblock2 = ResidualBlockWithAttention(160)

    def forward(self, x):
        x = self.resblock1(x)
        x = self.up1(x)  # 上采样到 [B, 160, 23, 23]
        x = self.resblock2(x)
        return x


def AC_loss(teacher, student):

    teacher_norm = teacher.norm(p=2, dim=2)
    student_norm = student.norm(p=2, dim=2)
    teacher_norm = torch.mean(teacher_norm, dim=2, keepdim=False)
    student_norm = torch.mean(student_norm, dim=2, keepdim=False)
    absdiff = torch.abs(teacher_norm - student_norm)

    cosineSimilarity = torch.nn.CosineSimilarity(dim=2, eps=1e-6)(teacher, student)
    cosineSimilarity = 1 - cosineSimilarity
    cosineSimilarity = torch.mean(cosineSimilarity, dim=2, keepdim=False)

    total = absdiff + cosineSimilarity
    C = 0.6 * torch.max(total).item()
    loss = torch.mean(torch.where(total < C, total, (total * total + C * C) / (2 * C)))

    return loss