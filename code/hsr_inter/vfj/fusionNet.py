import torch.nn as nn
import torch
import numpy as np

class FusionNet(nn.Module):
    '''
    network which has a spatial and temporal model ,an additional layers to fuse
    and finally some FC layers
    '''
    def __init__(self, spatial_model, forceTorque_model):
        super(FusionNet, self).__init__()
        self.spatial_model = spatial_model
        self.forceTorque_model = forceTorque_model
        self.fc1 = nn.Sequential(nn.Linear(512, 128, bias=True), nn.ReLU(inplace=True), nn.Dropout(p = 0.5))
        self.fc2 = nn.Sequential(nn.Linear(128+32, 32, bias=True), nn.ReLU(inplace=True), nn.Dropout(p = 0.5),
                                nn.Linear(32, 1))

    def forward(self, spatial_input, ft_input):
        spatial_output = self.spatial_model(spatial_input)
        spatial_output = self.fc1(spatial_output)

        ft_output = self.forceTorque_model(ft_input)

        # print('spatial shape:', spatial_output.shape)
        # print('ft shape:', ft_output.shape)
        fused = torch.cat((spatial_output, ft_output), dim = 1)
        fused = fused.view(fused.size(0), -1)
        fused = self.fc2(fused)


        # fused = torch.div((spatial_output + tactile_output + position_output),3.0)
        # fused = nn.functional.softmax(fused)
        # print('shape of fused= ', fused.shape)
        # out = fused.view(fused.size(0), -1)
        # print('shape of fused= ', fused.shape)
        # out = self.fc(out)
        # print('output of fc= ', out.shape)
        return fused
