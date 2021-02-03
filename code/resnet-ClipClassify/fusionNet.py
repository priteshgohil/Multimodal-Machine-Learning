import torch.nn as nn

class FusionNet(nn.Module):
    '''
    network which has a spatial and temporal model ,an additional layers to fuse
    and finally some FC layers
    '''
    def __init__(self, spatial_model):
        super(FusionNet, self).__init__()
        self.spatial_model = spatial_model
        self.fc = nn.Sequential(nn.Linear(512, 64, bias=True), nn.ReLU(inplace=True), nn.Dropout(p = 0.5),
                                nn.Linear(64, 1))

    def forward(self, spatial_input):
        spatial_output = self.spatial_model(spatial_input)
        #print('shape of fused= ', fused.shape)
        out = self.fc(spatial_output)
        #print('output of fc= ', out.shape)
        return out
