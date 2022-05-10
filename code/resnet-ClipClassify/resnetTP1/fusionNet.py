import torch.nn as nn
import torch

class FusionNet(nn.Module):
    '''
    network which has a spatial and temporal model ,an additional layers to fuse
    and finally some FC layers
    '''
    def __init__(self, tactile_model, position_model):
        super(FusionNet, self).__init__()
        self.tactile_model = tactile_model
        self.position_model = position_model
        self.fc = nn.Sequential(nn.Linear(64+64, 16, bias=True), nn.ReLU(inplace=True), nn.Dropout(p = 0.5),
                                nn.Linear(16, 1))

    def forward(self, tactile_input, position_input):
        tactile_output = self.tactile_model(tactile_input)
        position_output = self.position_model(position_input)

        fused = torch.cat((tactile_output, position_output), dim = 1)
        # print('shape of fused= ', fused.shape)
        out = fused.view(fused.size(0), -1)
        # print('shape of fused= ', fused.shape)
        out = self.fc(out)
        # print('output of fc= ', out.shape)
        return out
