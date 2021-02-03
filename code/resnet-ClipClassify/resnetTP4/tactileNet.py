import torch.nn as nn
class tactileNet(nn.Module):
    def __init__(self):
        super(tactileNet,self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(192+96, 64, bias=True), nn.ReLU(inplace=True), nn.Dropout(p = 0.5))
        self.fc2 = nn.Sequential(nn.Linear(64, 16, bias=True), nn.ReLU(inplace=True), nn.Dropout(p = 0.5))
        self.fc3 = nn.Linear(16,1)
    def forward(self, x):
        out = self.fc1(x)
        # print(out.shape)
        out = self.fc2(out)
        # print(out.shape)
        out = self.fc3(out)
        # print(out.shape)
        return out
