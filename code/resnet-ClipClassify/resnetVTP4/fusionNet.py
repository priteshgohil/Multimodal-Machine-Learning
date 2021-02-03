import torch.nn as nn
import torch

class FusionNet(nn.Module):
    '''
    network which has a spatial and temporal model ,an additional layers to fuse
    and finally some FC layers
    '''
    def __init__(self, spatial_model, tp_model):
        super(FusionNet, self).__init__()
        self.spatial_model = spatial_model
        self.tp_model = tp_model
        # self.fcResnet = nn.Sequential(nn.Linear(512, 64, bias=True), nn.ReLU(inplace=True), nn.Dropout(p = 0.5),
        #                                 nn.Linear(64, 1))
        self.fc = nn.Sequential(nn.Linear(512+16, 64, bias=True), nn.ReLU(inplace=True), nn.Dropout(p = 0.5),
                                            nn.Linear(64, 1))

    def forward(self, spatial_input, tp_input):
        spatial_output = self.spatial_model(spatial_input)
        # spatial_output = self.fcResnet(spatial_output)
        tp_output = self.tp_model(tp_input)

        fused = torch.cat((spatial_output, tp_output), dim = 1)
        # print('shape of fused= ', fused.shape)
        out = fused.view(fused.size(0), -1)
        # print('shape of fused= ', fused.shape)
        out = self.fc(out)
        # print('output of fc= ', out.shape)
        return out
