import torch.nn as nn
class tactileNet(nn.Module):
    def __init__(self):
        super(tactileNet,self).__init__()
        self.layer1 = nn.Sequential(nn.Conv3d(1, 16, kernel_size=(3,3,3), padding=2),nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv3d(16, 32, kernel_size=(3,3,3), padding=1),nn.ReLU(),nn.MaxPool3d(kernel_size=(3,3,3)))
        self.fc1 = nn.Sequential(nn.Linear(768, 64, bias=True), nn.ReLU(inplace=True), nn.Dropout(p = 0.5))
        self.fc2 = nn.Linear(64,1)
    def forward(self, x):
        # print(x.shape)
        out = self.layer1(x)
        # print(out.shape)
        out = self.layer2(out)
        # print(out.shape)
        out = out.view(out.size(0), -1)
        # print(out.shape)
        out = self.fc1(out)
        # print(out.shape)
        out = self.fc2(out)
        return out
