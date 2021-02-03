import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-mode', required=False, type=str, help='rgb or flow')
parser.add_argument('-save_model',required=True, type=str)
parser.add_argument('-root', required=True, type=str)
parser.add_argument('-test_dir', required=True, type=str)
parser.add_argument('-train_dir', required=True, type=str)
parser.add_argument('-save_filename', required=True, type=str)

args = parser.parse_args()

import tensorboardX
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms


import numpy as np

# from charades_dataset import Charades as Dataset
from visual_tactile_dataset import VisualTactile as Dataset
import cv2
import matplotlib.pyplot as plt

class tactileNet(nn.Module):
	def __init__(self):
		super(tactileNet,self).__init__()
		self.layer1 = nn.Sequential(nn.Conv3d(1, 16, kernel_size=(2,3,3), padding=1),nn.ReLU())
		self.layer2 = nn.Sequential(nn.Conv3d(16, 32, kernel_size=(2,3,3), padding=0),nn.ReLU(),nn.MaxPool3d(kernel_size=(2,4,4)))
		self.fc1 = nn.Sequential(nn.Linear(1024, 64, bias=True), nn.ReLU(inplace=True), nn.Dropout(p = 0.5))
#         self.fc2 = nn.Linear(64,1)
	def forward(self, x):
		out = self.layer1(x)
#         print(out.shape)
		out = self.layer2(out)
#         print(out.shape)
		out = out.view(out.size(0), -1)
#         print(out.shape)
		out = self.fc1(out)
#         print(out.shape)
		return out

class tactileFusionNet(nn.Module):
	def __init__(self, tac_model1, tac_model2, tac_model3, tac_model4):
		super(tactileFusionNet,self).__init__()
		self.tac_model1 = tac_model1
		self.tac_model2 = tac_model2
		self.tac_model3 = tac_model3
		self.tac_model4 = tac_model4
		self.fc1 = nn.Sequential(nn.Linear(64*4, 128, bias=True), nn.ReLU(inplace=True), nn.Dropout(p = 0.5))
		self.fc2 = nn.Linear(128,1)
	def forward(self, x1,x2,x3,x4):
		t1 = self.tac_model1(x1)
		t2 = self.tac_model2(x2)
		t3 = self.tac_model3(x3)
		t4 = self.tac_model4(x4)
		out = torch.cat((t1,t2,t3,t4),dim=1)
		out = self.fc1(out)
#         print(out.shape)
		out = self.fc2(out)
#         print(out.shape)
		return out

def save_checkpoint(model, optimizer, scheduler, epoch,save_model):
	data = {'model_state' : model.state_dict(),
            'optimizer_state' : optimizer.state_dict(),
            'scheduler_state' : scheduler.state_dict(),
            'epoch' : epoch+1}
	torch.save(data, 'model_%d' %(epoch + 1) +save_model + '.tar' )

def load_model(learning_rate, scheduler_list, checkpoint=None):
    sm = InceptionI3d(400, in_channels=3)
    sm.replace_logits(1)
    fusedNet = FusionNet(sm)
    #i3d = i3d.to(self.device)
    optimizer = optim.SGD(fusedNet.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0000001)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, scheduler_list)
    if (checkpoint is not None):
        data = torch.load(checkpoint)
        fusedNet.load_state_dict(data['model_state'])
        optimizer.load_state_dict(data['optimizer_state'])
        scheduler.load_state_dict(data['scheduler_state'])
    return fusedNet, optimizer, scheduler
    

def freeze_network_layer(model):
    for para in model.parameters():
        para.requires_grad = False
    return model
# def run(init_lr=0.1, max_steps=64e3, mode='rgb', root='/ssd/Charades_v1_rgb', train_split='charades/charades.json', batch_size=8*5, save_model=''):
def run(init_lr=0.001, max_steps=30, mode='rgb', root='/media/pritesh/Entertainment/Visual-Tactile_Dataset/dataset/',\
        train_split='trainv2.txt', test_split='testv2.txt', batch_size=100, save_model='', save_filename=''):
	writer = tensorboardX.SummaryWriter()
    # setup dataset

	dataset = Dataset(train_split, root, mode)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=True)

	val_dataset = Dataset(test_split, root, mode)
	val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=True)

	dataloaders = {'train': dataloader, 'val': val_dataloader}
	datasets = {'train': dataset, 'val': val_dataset}


    # setup the model

    #add your network here
    # fusedNet = FusionNet(sm,tm)
	fuseNet1 = tactileNet()
	fuseNet2 = tactileNet()
	fuseNet3 = tactileNet()
	fuseNet4 = tactileNet()
	masterNet = tactileFusionNet(fuseNet1,fuseNet2,fuseNet3,fuseNet4)
	if torch.cuda.is_available():
		fuseNet1.cuda()
		fuseNet2.cuda()
		fuseNet3.cuda()
		fuseNet4.cuda()
		masterNet.cuda()
	fusedNet1 = nn.DataParallel(fuseNet1)
	fusedNet2 = nn.DataParallel(fuseNet2)
	fusedNet3 = nn.DataParallel(fuseNet3)
	fusedNet4 = nn.DataParallel(fuseNet4)
	masterNet = nn.DataParallel(masterNet)
	lr = init_lr
	optimizer = optim.SGD(masterNet.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
	#optimizer = optim.Adam(masterNet.parameters(), lr=lr, weight_decay=0.0000001)
	lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [10,20,25])
    
	steps = 0
	with open(save_filename, 'w') as file:
		file.write("train and validation loss file\n")
    # train it
	while steps < max_steps:#for epoch in range(num_epochs):
		print ('Step {}/{}'.format(steps, max_steps))
		print ('-' * 10)

        # Each epoch has a training and validation phase
		for phase in ['train', 'val']:
			print('phase : {}'.format(phase))
			if phase == 'train':
				masterNet.train(True)
			else:
				masterNet.train(False)  # Set model to evaluate mode

			tot_cls_loss = 0.0
			num_iter = 0
			count = 0
			optimizer.zero_grad()
            # Iterate over data.
			for data in dataloaders[phase]:
				num_iter += 1
        	    # get the inputs
				tactile, labels = data

				if torch.cuda.is_available():
					t1 = Variable(tactile[0].cuda())
					t2 = Variable(tactile[1].cuda())
					t3 = Variable(tactile[2].cuda())
					t4 = Variable(tactile[3].cuda())
					labels = Variable(labels.cuda())
				else:
					t1 = Variable(tactile[0])
					t2 = Variable(tactile[1])
					t3 = Variable(tactile[2])
					t4 = Variable(tactile[3])
					labels = Variable(labels)
                
				out = masterNet(t1.float(),t2.float(),t2.float(),t2.float())
                #print('prediction output = ', per_frame_logits.shape)
                #print('labels = ',labels.shape)
                # compute classification loss (with max-pooling along time B x C x T)
				out = out.squeeze(1)
				cls_loss = F.binary_cross_entropy_with_logits(out.double(), labels.double())
				tot_cls_loss += cls_loss.item()
				cls_loss.backward()
				print('{} Loss: {:.4f} and lr: {}'.format(phase,tot_cls_loss/num_iter,init_lr))
				with open(save_filename, 'a') as file:
					file.write('epoch: ({}) {}: {} Loss: {:.7f} \n'.format(steps+1, num_iter,phase,tot_cls_loss/num_iter))
				optimizer.step()
				optimizer.zero_grad()
				if phase == 'val':
					writer.add_scalar('error/'+ phase , (tot_cls_loss/num_iter), num_iter)
				else:
					writer.add_scalar('error/'+ phase, (tot_cls_loss/num_iter), num_iter)
					if((steps+1)%30 == 0):
						save_checkpoint(masterNet, optimizer, lr_sched, steps,save_model)
			#save error at every epoch
			writer.add_scalar('errorAtEpoch/'+phase, (tot_cls_loss/num_iter), steps)
			tot_cls_loss = 0.
			with open(save_filename, 'a') as file:
				file.write("-"*50)
				file.write("\n")
        #if(steps%50 == 0):
        #    torch.save(fusedNet.module.state_dict(), save_model+phase+str(steps).zfill(6)+'.pt')
        #    save_checkpoint(fusedNet, optimizer, lr_sched, steps)
		steps+=1
		lr_sched.step()


if __name__ == '__main__':
	run(root=args.root, train_split=args.train_dir, test_split=args.test_dir, save_model=args.save_model, save_filename=args.save_filename)
