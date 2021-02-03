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
import videotransforms


import numpy as np

from pytorch_i3d import InceptionI3d
from pytorch_i3d import FusionNet

# from charades_dataset import Charades as Dataset
from visual_tactile_dataset_v import VisualTactile as Dataset
import cv2
import matplotlib.pyplot as plt


def run(init_lr=0.01, max_steps=200, mode='rgb', root='/media/pritesh/Entertainment/Visual-Tactile_Dataset/dataset/',\
        train_split='train.txt', test_split='test.txt', batch_size=1, save_model='',save_file=''):
    print(train_split, test_split)
    writer = tensorboardX.SummaryWriter(comment="test")
    # setup dataset
    test_transforms = transforms.Compose([videotransforms.CenterCrop(112)])

    dataset = Dataset(train_split, root, mode, test_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=True)

    val_dataset = Dataset(test_split, root, mode, test_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=True)

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
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [10,20,25])
    if torch.cuda.is_available():
        data = torch.load(save_model)
    else:
        data = torch.load(save_model, map_location=lambda storage, loc: storage)
    fusedNet.load_state_dict(data['model_state'])
    optimizer.load_state_dict(data['optimizer_state'])
    lr_sched.load_state_dict(data['scheduler_state'])

    steps = 0
    with open(save_file, 'w') as file:
        file.write("train and validation loss file\n")
    # train it
    # Each epoch has a training and validation phase

    fusedNet.train(False)  # Set model to evaluate mode
    test_TP,test_TN,test_FP,test_FN, train_TP,train_TN,train_FP,train_FN = 0,0,0,0,0,0,0,0
    actual_out, predicted_out = 0,0
    for phase in ['train', 'val']:
        print('phase : {}'.format(phase))

        tot_cls_loss = 0.0
        num_iter = 0
        count = 0
#         optimizer.zero_grad()

        with open(save_file, 'a') as file:
                file.write("---------------\n")
        # Iterate over data.
        for data in dataloaders[phase]:
            num_iter += 1
            # get the inputs
            f_vid, l_vid, tactile, pos, labels, directory = data

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
            actual_out = labels[0]
            predicted_out = 1 if out[0]>0 else 0
#            if(out[0]>0):
#                predicted_out = 1
#            else:
#                predicted_out = 0
#             cls_loss.backward()
            print('{}: {} Loss: {:.5f} and lr: {:.5f}.... Network output: {:.5f} and actual label: {} directory: {}'.format(num_iter,phase,cls_loss.item(),lr_sched.get_lr()[0],out[0], labels[0], directory))
            with open(save_file, 'a') as file:
                file.write('{}: {} Loss: {:.5f} and lr: {:.5f}.... Network output: {:.5f} and actual label: {} directory: {}\n'.format(num_iter,phase,cls_loss.item(),lr_sched.get_lr()[0],out[0], labels[0], directory))
#             optimizer.step()
#             optimizer.zero_grad()
            if phase == 'val':
                writer.add_scalar('inference_error/'+ phase , (cls_loss.item()), num_iter)
                if actual_out == predicted_out:
                    if actual_out:
                        test_TP +=1
                    else:
                        test_TN +=1
                else:
                    if actual_out:
                        test_FN +=1
                    else:
                        test_FP +=1
            else:
                writer.add_scalar('inference_error/'+ phase, (cls_loss.item()), num_iter)
                if actual_out == predicted_out:
                    if actual_out:
                        train_TP +=1
                    else:
                        train_TN +=1
                else:
                    if actual_out:
                        train_FN +=1
                    else:
                        train_FP +=1

    with open(save_file, 'a') as file:
        file.write("-"*100)
        file.write("\n")
        file.write("Network information\n")
        file.write("learning rate: {}, epoch: {}, batch size: {}, saved model name: {}, optimizer : SGD, learning steps: [10,20,25] \n".format(init_lr,max_steps,batch_size,save_model))
        file.write("-"*100)
        file.write("\n")
        file.write("Confusion matrix for training data\n")
        file.write("TP: {}, TN: {}, FP: {}, FN: {} \n".format(train_TP, train_TN, train_FP, train_FN))
        file.write("-"*100)
        file.write("\n")
        file.write("Confusion matrix for test data \n")
        file.write("TP: {}, TN: {}, FP: {}, FN: {} \n".format(test_TP, test_TN, test_FP, test_FN))
    #if(steps%50 == 0):
    #    torch.save(fusedNet.module.state_dict(), save_model+phase+str(steps).zfill(6)+'.pt')
    #    save_checkpoint(fusedNet, optimizer, lr_sched, steps)

if __name__ == '__main__':
    run(root=args.root, train_split=args.train_dir, test_split=args.test_dir, save_model=args.save_model, save_file=args.save_filename)
