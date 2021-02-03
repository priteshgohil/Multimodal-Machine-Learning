import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
import sys
import argparse
import tensorboardX
import videotransforms
import torchvision
from torchvision import datasets, transforms
from dataset import VisualTactile
from fusionNet import FusionNet
import torch.nn as nn
import torch
import resnet
import tactileNet
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.nn.functional as F
import time

parser = argparse.ArgumentParser()
parser.add_argument('-root', required=True, type=str, help = 'dataset path')
parser.add_argument('-mode', required=True, type=str, help='specify train or test')
parser.add_argument('-test_dir', required=True, type=str, help = 'test.txt file path')
parser.add_argument('-train_dir', required=True, type=str, help = 'train.txt file path')
parser.add_argument('-save_model',required=False, type=str, help='enter the name you would like to save model with')
parser.add_argument('-save_error', required=True, type=str)
parser.add_argument('-checkpoint', required=False, type=str, help='path to saved model')

args = parser.parse_args()

class Run_Model(object):

    def __init__(self, root, mode, test_dir, train_dir, save_model_withname=None,\
                 save_error_withname=None, checkpoint=None):
        self.root = root
        self.mode = mode
        self.test_dir = test_dir
        self.train_dir = train_dir
        self.save_model_withname = save_model_withname
        self.save_error_withname = save_error_withname
        self.checkpoint = checkpoint
        self.batch_size = 50
        self.learning_rate = 0.0001
        self.validation_loop = 0

        if(self.mode=='train'):
            self.writer = tensorboardX.SummaryWriter(comment="train")
        else:
            self.writer = tensorboardX.SummaryWriter(comment="test")
        # setup dataset
        self.train_transforms = transforms.Compose([videotransforms.RandomCrop(112),
                                           videotransforms.RandomHorizontalFlip(),])
        self.test_transforms = transforms.Compose([videotransforms.CenterCrop(112)])

        self.dataset = VisualTactile(self.root, self.train_dir, self.train_transforms)
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=1, pin_memory=True)

        self.val_dataset = VisualTactile(self.root, self.test_dir, self.test_transforms)
        self.val_dataloader = torch.utils.data.DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

#         self.dataloaders = {'train': self.dataloader, 'val': self.val_dataloader}
#         self.datasets = {'train': self.dataset, 'val': self.val_dataset}

        self.model, self.optimizer, self.scheduler = self.load_model(self.checkpoint)

    def load_model(self, checkpoint):
#         sm = resnet.resnet18(sample_size = 112, sample_duration = 18, num_classes = 400, shortcut_type='A')
#         sm = nn.DataParallel(sm)
#         if torch.cuda.is_available():
#             pretrain = torch.load("../models/resnet-18-kinetics.pth")
#             sm.load_state_dict(pretrain['state_dict'])
#         else:
#             pretrain = torch.load("../models/resnet-18-kinetics.pth", map_location="cpu")
# #            pretrain = torch.load("../../../out/resnet/resnet-18-kinetics.pth", map_location="cpu")
#             sm.load_state_dict(pretrain['state_dict'])
#         sm = self.freeze_network_layer(sm)
        net = tactileNet.tactileNet()
        if torch.cuda.is_available():
            net.cuda()
            # net = nn.DataParallel(net)
        optimizer = optim.Adam(net.parameters(), lr=self.learning_rate, weight_decay=0.0000001)
        lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [10, 20, 25])

        #checkpoint in case of training
        if (checkpoint is not None):
            if torch.cuda.is_available():
                data = torch.load(checkpoint)
            else:
                data = torch.load(checkpoint, map_location=lambda storage, loc: storage)
            net.load_state_dict(data['model_state'])
            optimizer.load_state_dict(data['optimizer_state'])
            lr_sched.load_state_dict(data['scheduler_state'])
        return net, optimizer, lr_sched

    def freeze_network_layer(self, model):
        for para in model.parameters():
            para.requires_grad = False
        return model

    def save_checkpoint(self, model, optimizer, scheduler, epoch, save_model):
        data = {'model_state' : model.state_dict(),
                'optimizer_state' : optimizer.state_dict(),
                'scheduler_state' : scheduler.state_dict(),
                'epoch' : epoch+1}
        torch.save(data, 'model_%d' %(epoch + 1) +save_model + '.tar' )

    def train(self):
        with open(self.save_error_withname, 'w') as file:
            file.write("train loss file\n")
        epoch_num = 30
        self.model.train(True)
        for epoch in range(epoch_num):
            print ('Step {}/{}'.format(epoch, epoch_num))
            print ('-' * 10)
            epoch_time = time.time()

            self.optimizer.zero_grad()
            total_loss = 0
            for i,data in enumerate(self.dataloader):
                tp_early_fused, lab, path = data

                if torch.cuda.is_available():
                    tp_fused = Variable(tp_early_fused.cuda())
                    label = Variable(lab.cuda())
                else:
                    tp_fused = Variable(tp_early_fused)
                    label = Variable(lab)

                out = self.model(tp_fused.float())
                out = out.squeeze(1)

                loss = F.binary_cross_entropy_with_logits(out.float(), label.float())
                total_loss += loss.item()
                loss.backward()
                print('{} Loss: {:.4f} and lr: {}'.format(self.mode,total_loss/(i+1),self.scheduler.get_lr()[0]))
                with open(self.save_error_withname, 'a') as file:
                    file.write("epoch: {}, Loss: {}, LR: {}\n".format(epoch, total_loss/(i+1),self.scheduler.get_lr()[0]))

                self.optimizer.step()
                self.optimizer.zero_grad()
                self.writer.add_scalar("error/{}".format(epoch), total_loss/(i+1), i)
            self.writer.add_scalar("errorPerEpoch/", total_loss/(i+1), epoch)
            self.writer.close()
            self.scheduler.step()
            print("epoch {} :: time {}".format(epoch,(time.time()-epoch_time)/60))
            if((epoch+1)%epoch_num == 0):
                self.save_checkpoint(self.model, self.optimizer, self.scheduler, epoch, self.save_model_withname)
            if((epoch+1)%5 == 0):
                self.validation_loop +=1
                self.test() #check validation error at every 5th epoch


    def test(self):
        if(self.validation_loop):
            with open(self.save_error_withname, 'a') as file:
                file.write("test loss file\n")
        else:
            with open(self.save_error_withname, 'w') as file:
                file.write("test loss file\n")
        self.model.train(False)
        test_TP,test_TN,test_FP,test_FN= 0,0,0,0
        actual_out, predicted_out = 0,0
        video_num = self.val_dataset.get_num_videos()
        count = 0
        total_loss = 0
        for index in range(video_num):
            data = self.val_dataset.get_video_frames(index)
            packed_data, vid_path = data
            print("directory: {}".format(vid_path))
            # iterating though mini clips in a video
            for i,dota in enumerate(packed_data):
                tp_early_fused, label = dota
                label = label.unsqueeze(0) #because without this its shape is empty
                if torch.cuda.is_available():
                    tp_fused = Variable(tp_early_fused.cuda())
                    label = Variable(label.cuda())
                else:
                    tp_fused = Variable(tp_early_fused)
                    label = Variable(label)

                tp_fused = tp_fused.unsqueeze(0)

                out = self.model(tp_fused.float())
                out = out.squeeze(1)

                loss = F.binary_cross_entropy_with_logits(out.float(), label.float())
                total_loss += loss.item()
                actual_out = label[0]
                predicted_out = 1 if out[0]>0 else 0
                print('{}: Loss: {:.5f} and lr: {:.5f}.... Network output: {:.5f} and actual label: {} '.format(i,loss.item(),self.scheduler.get_lr()[0],out[0], label[0]))
                with open(self.save_error_withname, 'a') as file:
                    file.write('{}: Loss: {:.5f} and lr: {:.5f}.... Network output: {:.5f} and actual label: {}, total_loss: {} \n'.format(i,loss.item(),self.scheduler.get_lr()[0],out[0], label[0], total_loss/(count+1)))
                # self.writer.add_scalar('inference_error/{}'.format(index), loss.item(), i)
                self.writer.add_scalar('combined_inference_error/', loss.item(), count)
                count += 1

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
        self.writer.add_scalar('total validation loss: ',(total_loss/count), self.validation_loop)
        self.writer.close()
        with open(self.save_error_withname, 'a') as file:
            file.write("-"*100)
            file.write("\n")
            file.write("Network information\n")
            file.write("learning rate: {}, batch size: {}, saved model name: {}, optimizer : Adam, learning steps: [10,20,25] \n".format(self.learning_rate,self.batch_size, self.save_model_withname))
            file.write("-"*100)
            file.write("\n")
            file.write("Confusion matrix for test data \n")
            file.write("TP: {}, TN: {}, FP: {}, FN: {} \n".format(test_TP, test_TN, test_FP, test_FN))
        self.model.train(True)

if __name__ == '__main__':
    trainer = Run_Model(root = args.root, mode = args.mode, test_dir=args.test_dir,
                       train_dir = args.train_dir, save_model_withname=args.save_model,
                       save_error_withname = args.save_error, checkpoint=args.checkpoint)

    if(args.mode == 'train'):
        trainer.train()
    else:
        trainer.test()
