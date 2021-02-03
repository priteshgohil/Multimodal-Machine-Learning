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
from pytorch_i3d import FusionNet

# from charades_dataset import Charades as Dataset
from visual_tactile_dataset import VisualTactile as Dataset
import cv2
import matplotlib.pyplot as plt


def save_checkpoint(model, optimizer, scheduler, epoch):
    data = {'model_state' : model.state_dict(),
            'optimizer_state' : optimizer.state_dict(),
            'scheduler_state' : scheduler.state_dict(),
            'epoch' : epoch+1}
    torch.save(data, 'model_vf_%d.tar' % (epoch + 1))

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

def freeze_network_layer(model):
    for para in model.parameters():
        para.requires_grad = False
    return model
# def run(init_lr=0.1, max_steps=64e3, mode='rgb', root='/ssd/Charades_v1_rgb', train_split='charades/charades.json', batch_size=8*5, save_model=''):
def run(init_lr=0.01, max_steps=200, mode='rgb', root='/media/pritesh/Entertainment/Visual-Tactile_Dataset/dataset/',\
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
    sm = InceptionI3d(400, in_channels=3)
    sm.load_state_dict(torch.load('models/rgb_imagenet.pt'))
    sm.replace_logits(1)
    sm = freeze_network_layer(sm)
    
    tm = InceptionI3d(400, in_channels=2, spatial_squeeze=False)
    tm.load_state_dict(torch.load('models/flow_imagenet.pt'))
    tm.replace_logits(1)
    tm = freeze_network_layer(tm)
    #add your network here
    fusedNet = FusionNet(sm,tm)
    if torch.cuda.is_available():
        fusedNet.cuda()
    fusedNet = nn.DataParallel(fusedNet)

    lr = init_lr
    optimizer = optim.SGD(fusedNet.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [50, 100, 150, 200])
    
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
                fusedNet.train(True)
            else:
                fusedNet.train(False)  # Set model to evaluate mode

            tot_cls_loss = 0.0
            num_iter = 0
            count = 0
            optimizer.zero_grad()
            # Iterate over data.
            for data in dataloaders[phase]:
                num_iter += 1
                # get the inputs
                f_vid, l_vid, tactile, pos, f_flow, l_flow, labels = data

                if torch.cuda.is_available():
                    rgb_inputs = Variable(f_vid.cuda())
                    flow_inputs = Variable(f_flow.cuda())
                    t = rgb_inputs.size(2)
                    labels = Variable(labels.cuda())
                else:
                    rgb_inputs = Variable(f_vid)
                    flow_inputs = Variable(f_flow)
                    t = rgb_inputs.size(2)
                    labels = Variable(labels)
                
                out = fusedNet(rgb_inputs.float(), flow_inputs.float())
                #print('prediction output = ', per_frame_logits.shape)
                #print('labels = ',labels.shape)
                # compute classification loss (with max-pooling along time B x C x T)
                out = out.squeeze(1)
                cls_loss = F.binary_cross_entropy_with_logits(out.double(), labels.double())
                tot_cls_loss += cls_loss.item()
                cls_loss.backward()
                print('{} Loss: {:.4f} and lr: {}'.format(phase,tot_cls_loss/num_iter,init_lr))
                with open('i3d_video.txt', 'a') as file: 
                    file.write("%f\n" %(tot_cls_loss/num_iter))
                optimizer.step()
                optimizer.zero_grad()
                if phase == 'val':
                    writer.add_scalar('error/'+ phase , (tot_cls_loss/num_iter), num_iter)
                else:
                    writer.add_scalar('error/'+ phase, (tot_cls_loss/num_iter), num_iter)
                    if(steps%50 == 0):
                        torch.save(fusedNet.module.state_dict(), save_model+phase+str(steps).zfill(6)+'.pt')
                        save_checkpoint(fusedNet, optimizer, lr_sched, steps)
            #save error at every epoch
            writer.add_scalar('errorAtEpoch/'+phase, (tot_cls_loss/num_iter), steps)
            tot_cls_loss = 0.
        #if(steps%50 == 0):
        #    torch.save(fusedNet.module.state_dict(), save_model+phase+str(steps).zfill(6)+'.pt')
        #    save_checkpoint(fusedNet, optimizer, lr_sched, steps)
        steps+=1
        lr_sched.step()


if __name__ == '__main__':
    run(root=args.root, train_split=args.train_dir, test_split=args.test_dir, save_model=args.save_model)
