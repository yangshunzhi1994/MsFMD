#!/usr/bin/python3
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from datasets.RAF import RAF
from datasets.ExpW import ExpW
from datasets.CK_Plus import CK_Plus
from teacherNet import Teacher
from studentNet import CNN_RIS
import utils
from utils import load_pretrained_model
import losses
import itertools
from itertools import chain
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='train kd')

# various path
parser.add_argument('--save_root', type=str, default='results/', help='models and logs are saved here')
parser.add_argument('--t_model', type=str, default="Teacher", help='name of teacher model')
parser.add_argument('--s_model', type=str, default="CNNRIS", help='name of student model')
parser.add_argument('--distillation', type=str, default="Full", help='DE1, DE2, Full')
parser.add_argument('--data_name', type=str, default='ExpW', help='ExpW,RAF,CK_Plus')

# training hyper parameters
parser.add_argument('--epochs', type=int, default=240, help='number of total epochs to run')
parser.add_argument('--train_bs', default=128, type=int, help='learning rate')  # 32
parser.add_argument('--test_bs', default=256, type=int, help='learning rate')
parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate') # 0.01
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')#1e-4,5e-4
parser.add_argument('--cuda', type=int, default=1)
parser.add_argument('--seed', type=int, default=0, help='random seed')

parser.add_argument('-c', '--alpha', type=float, default=0.2, help='weight for classification')
parser.add_argument('-k', '--beta', type=float, default=0.8, help='weight balance for KD')
parser.add_argument('-o', '--gamma', type=float, default=2.0, help='weight balance for other losses')
parser.add_argument('-d', '--delta', type=float, default=2.0, help='weight balance for other losses')
parser.add_argument('--S_size', default=44, type=int, help='44,32,24,16,8')
parser.add_argument('--noise', type=str, default='none', help='GaussianBlur,AverageBlur,MedianBlur,BilateralFilter,Salt-and-pepper')

args, unparsed = parser.parse_known_args()

path = os.path.join(args.save_root + args.data_name+ '_' + args.t_model + '_' + args.s_model
					+ '_'+ args.distillation + '_c_' + str(args.alpha) + '_k_' + str(args.beta) + '_o_'
					+ str(args.gamma) + '_d_' + str(args.delta) + '_d_' + str(args.seed) + '_' + str(args.noise) + '_' + str(args.S_size))
text_path = path + '/' + args.data_name+ '_' + args.t_model + '_' + args.s_model + '_'+ args.distillation + '_c_' \
			+ str(args.alpha) + '_k_' + str(args.beta) + '_o_' + str(args.gamma) + '_d_' + str(args.delta) + '_' + str(args.noise)
writer = SummaryWriter(log_dir=path)

torch.cuda.manual_seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed(args.seed)

if args.s_model == 'CNNRIS':
	snet = CNN_RIS()
else:
	raise Exception('Invalid name of the student network...')

if args.t_model == 'Teacher':
	tnet = Teacher()
else:
	raise Exception('Invalid name of the teacher network...')
tcheckpoint = torch.load(os.path.join(args.save_root + args.data_name+ '_' + args.t_model+ '_False', 'Best_Teacher_model.t7'))
load_pretrained_model(tnet, tcheckpoint['tnet'])
try:
	f = open(text_path, 'a')
	f.write('\nbest_Teacher_acc is '+ str(tcheckpoint['test_acc']))
except:
	f = open(text_path, 'a')
	f.write('\nbest_Teacher_acc is '+ str(tcheckpoint['best_PrivateTest_acc']))
f.write('\nThe dataset used for training is: '+ str(args.data_name))
f.write('\nThe distillation method is: '+ str(args.distillation))
f.write('\nThe type of noise used is:  '+ str(args.noise))
f.close()

tnet.eval()
for param in tnet.parameters():
	param.requires_grad = False

Cls_crit = torch.nn.CrossEntropyLoss().cuda()   #Classification loss
MSE_crit = nn.MSELoss().cuda() #MSE
KD_T_crit = losses.KL_divergence(temperature=20).cuda() #KL
tnet.cuda()
snet.cuda()
if args.distillation == 'DE1':
	decoder1 = losses.Decoder1().cuda()
	optimizer = torch.optim.SGD(itertools.chain(snet.parameters(), decoder1.parameters()), lr=args.lr,
								momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
	decoder1.train()
elif args.distillation == 'DE2':
	decoder2 = losses.Decoder2().cuda()
	optimizer = torch.optim.SGD(itertools.chain(snet.parameters(), decoder2.parameters()), lr=args.lr,
								momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
	decoder2.train()
elif args.distillation == 'Full':
	decoder1 = losses.Decoder1().cuda()
	decoder2 = losses.Decoder2().cuda()
	optimizer = torch.optim.SGD(itertools.chain(snet.parameters(), decoder1.parameters(), decoder2.parameters()),
								lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
	decoder1.train()
	decoder2.train()
else:
	raise Exception('Invalid distillation name...')

# define transforms
transform_train = transforms.Compose([
	transforms.RandomCrop(92),
	transforms.RandomHorizontalFlip(),
])

if args.data_name == 'RAF':
	transforms_teacher_Normalize = transforms.Normalize((0.5884594, 0.45767313, 0.40865755), 
                            (0.25717735, 0.23602168, 0.23505741))
	transforms_student_Normalize =  transforms.Normalize((0.58846486, 0.45766878, 0.40865615), 
                            (0.2516557, 0.23020789, 0.22939532))
	transforms_test_Normalize = transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(
            mean=[0.58846486, 0.45766878, 0.40865615], std=[0.2516557, 0.23020789, 0.22939532])
            (transforms.ToTensor()(crop)) for crop in crops]))

elif args.data_name == 'ExpW':
	transforms_teacher_Normalize = transforms.Normalize((0.5543365, 0.43664238, 0.3852607), 
                            (0.2749875, 0.24763328, 0.24209002))
	transforms_student_Normalize =  transforms.Normalize((0.55435663, 0.43664038, 0.38523868), 
                            (0.2706023, 0.24304168, 0.23759492))
	transforms_test_Normalize = transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(
            mean=[0.55435663, 0.43664038, 0.38523868], std=[0.2706023, 0.24304168, 0.23759492])
            (transforms.ToTensor()(crop)) for crop in crops]))

elif args.data_name == 'CK_Plus':
	transforms_teacher_Normalize = transforms.Normalize((0.59522575, 0.59511113, 0.5951069), 
                            (0.2783333, 0.27831876, 0.27831432))
	transforms_student_Normalize =  transforms.Normalize((0.5953254, 0.5952113, 0.59520704), 
                            (0.2709213, 0.27090552, 0.27090067))
	transforms_test_Normalize = transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(
            mean=[0.5953254, 0.5952113, 0.59520704], std=[0.2709213, 0.27090552, 0.27090067])
            (transforms.ToTensor()(crop)) for crop in crops]))

else:
	raise Exception('Invalid dataset name...')

teacher_norm = transforms.Compose([
transforms.ToTensor(),
transforms_teacher_Normalize,
])

student_norm = transforms.Compose([
transforms.Resize(args.S_size),
transforms.Resize(44),
transforms.RandomApply([
	transforms.Lambda(lambda img: utils.color(img, magnitude=0.5)),
	transforms.Lambda(lambda img: utils.posterize(img, magnitude=4)),
	transforms.Lambda(lambda img: utils.solarize(img, magnitude=128)),
	transforms.Lambda(lambda img: utils.contrast(img, magnitude=0.5)),
	transforms.Lambda(lambda img: utils.sharpness(img, magnitude=0.5)),
	transforms.Lambda(lambda img: utils.brightness(img, magnitude=0.5)),
	transforms.Lambda(lambda img: utils.autocontrast(img))
], p=0.3),  # 30%的概率随机应用某些增强操作
transforms.ToTensor(),
utils.Cutout(n_holes=1, length=13),
transforms_student_Normalize,
])

transform_test = transforms.Compose([
transforms.Resize(args.S_size),       # Downsample
transforms.Resize(44),       # Upsample
transforms.TenCrop(44),
transforms_test_Normalize,
])

if args.data_name == 'RAF':
	trainset = RAF(split = 'Training', transform=transform_train, student_norm=student_norm, teacher_norm=teacher_norm)
	PrivateTestset = RAF(split = 'PrivateTest', transform=transform_test, student_norm=None, teacher_norm=None, noise=args.noise)
elif args.data_name == 'ExpW':
	trainset = ExpW(split = 'Training', transform=transform_train, student_norm=student_norm, teacher_norm=teacher_norm)
	PrivateTestset = ExpW(split = 'PrivateTest', transform=transform_test, student_norm=None, teacher_norm=None)
elif args.data_name == 'CK_Plus':
	trainset = CK_Plus(split = 'Training', transform=transform_train, student_norm=student_norm, teacher_norm=teacher_norm)
	PrivateTestset = CK_Plus(split = 'PrivateTest', transform=transform_test, student_norm=None, teacher_norm=None, is_ATD=False, noise=args.noise)
else:
	raise Exception('Invalid dataset name...')

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_bs, shuffle=True, num_workers=8)
PrivateTestloader = torch.utils.data.DataLoader(PrivateTestset, batch_size=args.test_bs, shuffle=False, num_workers=8)

NUM_CLASSES = 7
best_acc = 0
def train(epoch):
	f = open(text_path, 'a')
	f.write('\n\nEpoch: %d' % epoch)
	snet.train()
	train_loss = 0
	train_cls_loss = 0
	sconf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES))
	current_lr = utils.adjust_learning_rate_wram_up(epoch, args.lr, lr_decay_rate=0.1, lr_decay_epochs=[150, 180, 210], optimizer=optimizer, wram_up=20)
	f.write('\nlearning_rate: %s' % str(current_lr))
	f.close()
	for batch_idx, (img_teacher, img_student, target) in enumerate(trainloader):

		if args.cuda:
			img_teacher = img_teacher.cuda()
			img_student = img_student.cuda()
			target = target.cuda()

		optimizer.zero_grad()
		
		img_teacher, img_student, target = Variable(img_teacher), Variable(img_student), Variable(target)

		rb1_s, rb2_s, rb3_s, _, out_s = snet(img_student)
		rb1_t, rb2_t, rb3_t, _, out_t = tnet(img_teacher)

		cls_loss = Cls_crit(out_s, target)
		kd_loss = KD_T_crit(out_t, out_s)

		if args.distillation == 'DE1':
			new_rb1_t, H_image  = decoder1(rb1_s)
			loss = args.alpha*cls_loss + args.beta*kd_loss + args.gamma*losses.styleLoss(img_teacher, H_image.cuda(), MSE_crit) + \
				   args.gamma*losses.styleLoss(rb1_t, new_rb1_t.cuda(), MSE_crit)
		elif args.distillation == 'DE2':
			block = decoder2(rb2_s)
			loss = args.alpha*cls_loss + args.beta*kd_loss + args.delta*losses.styleLoss(rb2_t, block, MSE_crit)
		elif args.distillation == 'Full':
			new_rb1_t, H_image  = decoder1(rb1_s)
			block = decoder2(rb2_s)
			loss = args.alpha*cls_loss + args.beta*kd_loss + args.gamma*losses.styleLoss(img_teacher, H_image.cuda(), MSE_crit) + \
				   args.gamma*losses.styleLoss(rb1_t, new_rb1_t.cuda(), MSE_crit) + args.delta*losses.styleLoss(rb2_t, block, MSE_crit)
		else:
			raise Exception('Invalid distillation name...')
		
		loss.backward()
		utils.clip_gradient(optimizer, 0.1)
		optimizer.step()
		train_loss += loss.item()
		train_cls_loss += cls_loss.item()

		sconf_mat, acc, mAP, F1_score = utils.ACC_evaluation(sconf_mat, out_s, target, NUM_CLASSES)

	return train_cls_loss/(batch_idx+1), 100.*acc, 100.* mAP, 100 * F1_score
	

def test(epoch):
	
	snet.eval()
	PrivateTest_loss = 0
	t_prediction = 0
	sconf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES))
	
	for batch_idx, (img, target) in enumerate(PrivateTestloader):
		t = time.time()
		test_bs, ncrops, c, h, w = np.shape(img)
		img = img.view(-1, c, h, w)
		if args.cuda:
			img = img.cuda()
			target = target.cuda()
		
		img, target = Variable(img), Variable(target)

		with torch.no_grad():
			rb1_s, rb2_s, rb3_s, _, out_s = snet(img)

		outputs_avg = out_s.view(test_bs, ncrops, -1).mean(1)

		loss = Cls_crit(outputs_avg, target)
		t_prediction += (time.time() - t)
		PrivateTest_loss += loss.item()

		sconf_mat, acc, mAP, F1_score = utils.ACC_evaluation(sconf_mat, outputs_avg, target, NUM_CLASSES)

	return PrivateTest_loss/(batch_idx+1), 100.*acc, 100.* mAP, 100 * F1_score

for epoch in range(1, args.epochs+1):
	# train one epoch
	train_loss, train_acc, train_mAP, train_F1 = train(epoch)
	# evaluate on testing set
	test_loss, test_acc, test_mAP, test_F1 = test(epoch)
	f = open(text_path, 'a')
	f.write("\ntrain_loss:  %0.3f, train_acc:  %0.3f, train_mAP:  %0.3f, train_F1:  %0.3f"%(train_loss, train_acc, train_mAP, train_F1))
	f.write("\ntest_loss:   %0.3f, test_acc:   %0.3f, test_mAP:   %0.3f, test_F1:   %0.3f"%(test_loss, test_acc, test_mAP, test_F1))
	f.close()
	writer.add_scalars('epoch/loss', {'train': train_loss, 'test': test_loss}, epoch)
	writer.add_scalars('epoch/accuracy', {'train': train_acc, 'test': test_acc}, epoch)
	writer.add_scalars('epoch/mAP', {'train': train_mAP, 'test': test_mAP}, epoch)
	writer.add_scalars('epoch/F1', {'train': train_F1, 'test': test_F1}, epoch)

	# save model
	if test_acc > best_acc:
		best_acc = test_acc
		best_mAP = test_mAP
		best_F1 = test_F1
		f = open(text_path, 'a')
		f.write('\nSaving models......')
		f.write("\nbest_PrivateTest_acc: %0.3f" % best_acc)
		f.write("\nbest_PrivateTest_mAP: %0.3f" % best_mAP)
		f.write("\nbest_PrivateTest_F1: %0.3f" % best_F1)
		f.close()
		state = {
			'epoch': epoch,
			'snet': snet.state_dict() if args.cuda else snet,
			'test_acc': test_acc,
			'test_mAP': test_mAP,
			'test_F1': test_F1,
			'test_epoch': epoch,
		} 
		torch.save(state, os.path.join(path, 'Student_Test_model.t7'))

f = open(text_path, 'a')
f.write("\n\n\nbest_PrivateTest_acc: %0.2f" % best_acc)
f.write("\nbest_PrivateTest_mAP: %0.2f" % best_mAP)
f.write("\nbest_PrivateTest_F1: %0.2f" % best_F1)
f.close()
