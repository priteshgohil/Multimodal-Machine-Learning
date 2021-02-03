import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-mode', required=False, type=str, help='rgb or flow')
parser.add_argument('-save_model',required=True, type=str)
parser.add_argument('-root', required=False, type=str)
parser.add_argument('-test_dir', required=False, type=str)
parser.add_argument('-train_dir', required=False, type=str)

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


def run(init_lr=0.01, max_steps=200, mode='rgb', root='/media/pritesh/Entertainment/Visual-Tactile_Dataset/dataset/',\
        train_split='train.txt', test_split='test.txt', batch_size=1, save_model=''):
    print(train_split, test_split)
    writer = tensorboardX.SummaryWriter()
    # setup dataset
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    dataset = Dataset(train_split, root, mode, test_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    val_dataset = Dataset(test_split, root, mode, test_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}


    # setup the model
    sm = InceptionI3d(400, in_channels=3)
    sm.replace_logits(1)
    #add your network here
    fusedNet = FusionNet(sm)
    if torch.cuda.is_available():
        fusedNet.cuda()
    fusedNet = nn.DataParallel(fusedNet)

    lr = init_lr
    optimizer = optim.SGD(fusedNet.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [50, 100, 150, 200])
    if torch.cuda.is_available():
        data = torch.load(save_model)
    else:
        data = torch.load(save_model, map_location=lambda storage, loc: storage)
    fusedNet.load_state_dict(data['model_state'])
    optimizer.load_state_dict(data['optimizer_state'])
    lr_sched.load_state_dict(data['scheduler_state'])
    
    steps = 0
    with open('inference_V.txt', 'w') as file: 
        file.write("train and validation loss file\n")
    # train it
    # Each epoch has a training and validation phase
    
    fusedNet.train(False)  # Set model to evaluate mode
    for phase in ['train', 'val']:
        print('phase : {}'.format(phase))

        tot_cls_loss = 0.0
        num_iter = 0
        count = 0
#         optimizer.zero_grad()
        
        with open('inference_V.txt', 'a') as file: 
                file.write("---------------\n")
        # Iterate over data.
        for data in dataloaders[phase]:
            num_iter += 1
            # get the inputs
            f_vid, l_vid, tactile, pos, labels = data

            if torch.cuda.is_available():
                rgb_inputs = Variable(f_vid.cuda())
                t = rgb_inputs.size(2)
                labels = Variable(labels.cuda())
            else:
                rgb_inputs = Variable(f_vid)
                t = rgb_inputs.size(2)
                labels = Variable(labels)

            out = fusedNet(rgb_inputs.float())
            #print('prediction output = ', per_frame_logits.shape)
            #print('labels = ',labels.shape)
            # compute classification loss (with max-pooling along time B x C x T)
            out = out.squeeze(1)
            cls_loss = F.binary_cross_entropy_with_logits(out.double(), labels.double())
            tot_cls_loss += cls_loss.item()
#             cls_loss.backward()
            print('{} Loss: {:.4f} and lr: {}'.format(phase,tot_cls_loss/num_iter,init_lr))
            with open('inference_V.txt', 'a') as file: 
                file.write("%f\n" %(tot_cls_loss/num_iter))
#             optimizer.step()
#             optimizer.zero_grad()
            if phase == 'val':
                writer.add_scalar('inference_error/'+ phase , (tot_cls_loss/num_iter), num_iter)
            else:
                writer.add_scalar('inference_error/'+ phase, (tot_cls_loss/num_iter), num_iter)
    #if(steps%50 == 0):
    #    torch.save(fusedNet.module.state_dict(), save_model+phase+str(steps).zfill(6)+'.pt')
    #    save_checkpoint(fusedNet, optimizer, lr_sched, steps)
    
if __name__ == '__main__':
    run(root=args.root, train_split=args.train_dir, test_split=args.test_dir, save_model=args.save_model)