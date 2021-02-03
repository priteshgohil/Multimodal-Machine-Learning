import torch.nn as nn
import torch

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
        self.fc = nn.Sequential(nn.Linear(512+64+64, 64, bias=True), nn.ReLU(inplace=True), nn.Dropout(p = 0.5),
                                nn.Linear(64, 1))

    def forward(self, spatial_input, tactile_input, position_input):
        spatial_output = self.spatial_model(spatial_input)
        tactile_output = self.tactile_model(tactile_input)
        position_output = self.position_model(position_input)

        fused = torch.cat((spatial_output, tactile_output, position_output), dim = 1)
        # print('shape of fused= ', fused.shape)
        out = fused.view(fused.size(0), -1)
        # print('shape of fused= ', fused.shape)
        out = self.fc(out)
        # print('output of fc= ', out.shape)
        return out