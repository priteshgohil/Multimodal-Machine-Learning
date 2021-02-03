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
import videotransforms


import numpy as np

from pytorch_i3d import InceptionI3d

# from charades_dataset import Charades as Dataset
from visual_tactile_dataset import VisualTactile as Dataset
import cv2
import matplotlib.pyplot as plt


def save_checkpoint(self, model, optimizer, scheduler, epoch):
    data = {'model_state' : model.state_dict(),
            'optimizer_state' : optimizer.state_dict(),
            'scheduler_state' : scheduler.state_dict(),
            'epoch' : epoch+1}
    torch.save(data, 'model_%d.tar' % (epoch + 1))

def load_model(input_channels, learning_rate, scheduler_list, checkpoint=None):
    i3d = InceptionI3d(400, in_channels=input_channels)
    #i3d = i3d.to(self.device)
    optimizer = optim.SGD(i3d.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0000001)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, scheduler_list)
    if (checkpoint is not None):
        data = torch.load(checkpoint)
        i3d.load_state_dict(data['model_state'])
        optimizer.load_state_dict(data['optimizer_state'])
        scheduler.load_state_dict(data['scheduler_state'])
    return i3d, optimizer, scheduler
    
def prediction(x):
    fc1 = nn.Linear(in_features=x.shape[1]*x.shape[2], out_features=1024)
    dropout = nn.Dropout(0.5)
    fc2 = nn.Linear(in_features=1024, out_features=1)
    fc1.cuda()
    fc2.cuda()
    dropout.cuda()
    
    x = x.view(-1,x.shape[1]*x.shape[2])
    x = fc1(x)
    x = fc2(dropout(x))
    return x

def freeze_network_layer(i3d):
    for para in i3d.parameters():
        para.requires_grad = False
    return i3d
# def run(init_lr=0.1, max_steps=64e3, mode='rgb', root='/ssd/Charades_v1_rgb', train_split='charades/charades.json', batch_size=8*5, save_model=''):
def run(init_lr=0.1, max_steps=200, mode='rgb', root='/media/pritesh/Entertainment/Visual-Tactile_Dataset/dataset/',\
        train_split='train.txt', test_split='test.txt', batch_size=5, save_model=''):
    writer = tensorboardX.SummaryWriter()
    # setup dataset
    train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
                                           videotransforms.RandomHorizontalFlip(),])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    dataset = Dataset(train_split, root, mode, train_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=True)

    val_dataset = Dataset(test_split, root, mode, test_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=True)

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}


    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load('models/flow_imagenet.pt'))
    else:
        i3d = InceptionI3d(400, in_channels=3)
        i3d.load_state_dict(torch.load('models/rgb_imagenet.pt'))
    i3d.replace_logits(1)
#     #i3d.load_state_dict(torch.load('/ssd/models/000920.pt'))
    i3d.cuda()
    i3d = nn.DataParallel(i3d)

    lr = init_lr
    optimizer = optim.SGD(i3d.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
#     lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [50, 100, 150, 200])


    num_steps_per_update = 4 # accum gradient
    steps = 0
    with open('i3d_video.txt', 'w') as file: 
        file.write("train and validation loss file\n")
    # train it
    while steps < max_steps:#for epoch in range(num_epochs):
        print ('Step {}/{}'.format(steps, max_steps))
        print ('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            print('phase : {}'.format(phase))
            if phase == 'train':
                i3d.train(True)
                i3d = freeze_network_layer(i3d)
            else:
                i3d.train(False)  # Set model to evaluate mode

            tot_loss = 0.0
            tot_loc_loss = 0.0
            tot_cls_loss = 0.0
            num_iter = 0
            count = 0
            optimizer.zero_grad()
            # Iterate over data.
            for data in dataloaders[phase]:
                num_iter += 1
                # get the inputs
                f_vid, l_vid, tactile, pos, labels = data

                # wrap them in Variable
                #inputs = Variable(f_vid)
                #t = inputs.size(2)
                #labels = Variable(labels)
                inputs = Variable(f_vid.cuda())
                t = inputs.size(2)
                labels = Variable(labels.cuda())
                
                x = i3d(inputs.float())
                per_frame_logits = prediction(x)
                #print('prediction output = ', per_frame_logits.shape)
                #print('labels = ',labels.shape)
                # compute classification loss (with max-pooling along time B x C x T)
                per_frame_logits = per_frame_logits.squeeze(1)
                cls_loss = F.binary_cross_entropy_with_logits(per_frame_logits.double(), labels.double())
                tot_cls_loss += cls_loss.item()
                cls_loss.backward()
                print('{} Loss: {:.4f} and lr: {}'.format(phase,tot_cls_loss/num_iter,init_lr))
                with open('i3d_video.txt', 'a') as file: 
                    file.write("%f\n" %(tot_cls_loss/num_iter))
                optimizer.step()
                optimizer.zero_grad()
                lr_sched.step()
                if phase == 'val':
                    writer.add_scalar('error/'+ phase , (tot_cls_loss/num_iter), num_iter)
                else:
                    writer.add_scalar('error/'+ phase, (tot_cls_loss/num_iter), num_iter)
            #save error at every epoch
            writer.add_scalar('errorAtEpoch/'+phase, (tot_cls_loss/num_iter), steps)
            tot_cls_loss = 0.
        if(steps%50 == 0):
            torch.save(i3d.module.state_dict(), save_model+phase+str(steps).zfill(6)+'.pt')
            save_checkpoint(i3d, optimizer, lr_sched, steps)
        steps+=1


if __name__ == '__main__':
    run(root=args.root, train_split=args.train_dir, test_split=args.test_dir, save_model=args.save_model)