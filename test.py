import pickle
import numpy as np
import pandas as pd
from PIL import Image

import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import models
import loader

from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig



def imprint(novel_loader, model):
    # switch to evaluate mode
    model.eval()
    print('Imprinting    ')
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(novel_loader):

            input = input.cuda()

            # compute output
            output = model.extract(input)

            if batch_idx == 0:
                output_stack = output
                target_stack = target
            else:
                output_stack = torch.cat((output_stack, output), 0)
                target_stack = torch.cat((target_stack, target), 0)
           
    
    new_weight = torch.zeros(50, 256)
    for i in range(50):
        tmp = output_stack[target_stack == (i + 50)].mean(0) 
        new_weight[i] = tmp / tmp.norm(p=2)
    weight = torch.cat((model.classifier.fc.weight.data, new_weight.cuda()))
    print(weight)
    model.classifier.fc = nn.Linear(256, 100, bias=False)
    model.classifier.fc.weight.data = weight


model = models.Net().cuda()
model_n = './pretrain_checkpoint/model_best.pth.tar'
checkpoint = torch.load(model_n)
model.load_state_dict(checkpoint['state_dict'])
print("=> loaded model checkpoint '{}' (epoch {})"
				.format(model_n, checkpoint['epoch']))

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
																	std=[0.5, 0.5, 0.5])
criterion = torch.nn.CrossEntropyLoss().cuda()

novel_dataset = loader.ImageLoader(
		'x', transforms.Compose([
				transforms.Resize(256),
				transforms.CenterCrop(224),
				transforms.ToTensor(),
				normalize,
		]),
		train=True, num_classes=100, 
		num_train_sample=20, 
		novel_only=True)

novel_loader = torch.utils.data.DataLoader(
		novel_dataset, batch_size=64, shuffle=False,
		num_workers=4, pin_memory=True)

train_dataset = loader.ImageLoader(
		'x', transforms.Compose([
				transforms.Resize(256),
				transforms.RandomCrop(224),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				normalize,
		]),
		train=True, num_classes=100, 
		num_train_sample=20)

train_loader = torch.utils.data.DataLoader(
		train_dataset, batch_size=64, sampler=train_dataset.get_balanced_sampler(),
		num_workers=4, pin_memory=True)

print(model.classifier)
imprint(novel_loader, model)
print(model.classifier)

for i, (input, target) in enumerate(train_loader):
	if i == 0:
		input = input.cuda()
		target = target.cuda(non_blocking=True)
		output = model(input)
		print("o::", output)
		print("t::", target)
		print("c::", criterion(output, target))


# with open('./cifar-100-python/test','rb') as fo:
# 	dict = pickle.load(fo, encoding='bytes')
# 	data_size = len(dict[b'fine_labels'])
# 	dict['imgs'] = []
# 	for i in range(data_size):
# 		tmp_img = dict[b'data'][i].reshape(3,32,32)
# 		im = Image.fromarray(np.swapaxes(np.swapaxes(tmp_img, 0, 2),0,1), 'RGB')
# 		dict['imgs'].append(im)
# 	df = pd.concat([pd.DataFrame(dict[b'filenames']),pd.DataFrame(dict[b'fine_labels'])\
# 		,pd.DataFrame(0,index=np.arange(data_size),columns=[2]),\
# 		pd.DataFrame(dict['imgs'])],axis=1)
# 	df.set_axis(['filenames','label','train_flag','data'],axis=1,inplace=True)
# 	print(df)

