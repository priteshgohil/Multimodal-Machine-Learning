import torch.nn as nn
import torch

class FusionNet(nn.Module):
    '''
    network which has a spatial and temporal model ,an additional layers to fuse
    and finally some FC layers
    '''
    def __init__(self, spatial_model, tactile_model):
        super(FusionNet, self).__init__()
        self.spatial_model = spatial_model
        self.tactile_model = tactile_model
        self.fc1 = nn.Sequential(nn.Linear(512, 128, bias=True), nn.ReLU(inplace=True), nn.Dropout(p = 0.5))
        self.fc2 = nn.Linear(128,1)

    def forward(self, spatial_input, tactile_input):
        spatial_output = self.spatial_model(spatial_input)
        spatial_output = self.fc1(spatial_output)
        spatial_output = self.fc2(spatial_output)

        tactile_output = self.tactile_model(tactile_input)

        # spatial_output = nn.functional.tanh(spatial_output)
        # tactile_output = nn.functional.tanh(tactile_output)

        fused = torch.div((spatial_output + tactile_output ),2.0)
        # print('shape of fused= ', fused.shape)
        # out = fused.view(fused.size(0), -1)
        # print('shape of fused= ', fused.shape)
        # out = self.fc(out)
        # print('output of fc= ', out.shape)
        return fused
