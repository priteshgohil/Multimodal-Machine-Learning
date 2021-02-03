import resnet
import torch
from torch import nn
import dataset

pretrained_path = "../../../out/resnet/resnet-18-kinetics.pth"
model = resnet.resnet18(sample_size = 112, sample_duration = 18, num_classes = 400, shortcut_type='A')

model = nn.DataParallel(model)
pretrain = torch.load(pretrained_path, map_location="cpu")
model.load_state_dict(pretrain['state_dict'])
