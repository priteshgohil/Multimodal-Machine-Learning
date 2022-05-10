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
        self.fc = nn.Linear(2,1)

    def forward(self, tactile_input, position_input):
        tactile_output = self.tactile_model(tactile_input)
        position_output = self.position_model(position_input)

        # tactile_output = nn.functional.tanh(tactile_output)
        # position_output = nn.functional.tanh(position_output)
        fused = torch.cat((tactile_output, position_output), dim = 1)
        #fused = (tactile_output + position_output)
        fused = fused.view(fused.size(0), -1)
        fused = self.fc(fused)
        # fused = nn.functional.sigmoid(fused)
        # nn.functional.sigmoid(out)
        # print(fused)
        # print('shape of fused= ', fused)
        # out = fused.view(fused.size(0), -1)
        # print('shape of fused= ', fused.shape)
        # out = self.fc(out)
        # print('output of fc= ', out.shape)
        return fused
