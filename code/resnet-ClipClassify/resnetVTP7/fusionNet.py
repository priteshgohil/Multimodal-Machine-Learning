# Author : Priteshkumar Gohil
import torch.nn as nn
import torch
import numpy as np

class FusionNet(nn.Module):
    '''
    network which has a spatial and temporal model ,an additional layers to fuse
    and finally some FC layers
    '''
    def __init__(self, spatial_model, tactile_model, position_model):
        super(FusionNet, self).__init__()
        self.spatial_model = spatial_model
        self.tactile_model = tactile_model
        self.position_model = position_model
        self.fc1 = nn.Sequential(nn.Linear(512, 128, bias=True), nn.ReLU(inplace=True), nn.Dropout(p = 0.5))
        self.fc2 = nn.Linear(128,1)
        self.fc3 = nn.Linear(3,1)

    def forward(self, spatial_input, tactile_input, position_input):
        spatial_output = self.spatial_model(spatial_input)
        spatial_output = self.fc1(spatial_output)
        spatial_output = self.fc2(spatial_output)

        tactile_output = self.tactile_model(tactile_input)
        position_output = self.position_model(position_input)

        # spatial_output = nn.functional.tanh(spatial_output)
        # tactile_output = nn.functional.tanh(tactile_output)
        # position_output = nn.functional.tanh(position_output)

        fused = torch.cat((spatial_output, tactile_output, position_output), dim = 1)
        fused = fused.view(fused.size(0), -1)
        fused = self.fc3(fused)


        # fused = torch.div((spatial_output + tactile_output + position_output),3.0)
        # fused = nn.functional.softmax(fused)
        # print('shape of fused= ', fused.shape)
        # out = fused.view(fused.size(0), -1)
        # print('shape of fused= ', fused.shape)
        # out = self.fc(out)
        # print('output of fc= ', out.shape)
        return fused
