import numpy
import torch.nn as nn

'''
Input shape : B x 6 x 18
Output shape: B x 16 x 2
'''
class WrenchNet(nn.Module):
    def __init__(self):
        super(WrenchNet, self).__init__()
        self.layer1 = nn.Conv1d(in_channels=6, out_channels=20, kernel_size=3, stride=2)
        self.act1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(20)
        self.layer2 = nn.Conv1d(in_channels=20, out_channels=30, kernel_size=3, stride=2)
        self.act2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(30)
        self.layer3 = nn.Conv1d(in_channels=30, out_channels=16, kernel_size=2)
        self.fc = nn.Linear(32,1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.act1(x)
        x = self.bn1(x)
        x = self.layer2(x)
        x = self.act2(x)
        x = self.bn2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        # x = self.fc(x)
        return x

# model = WrenchNet().double()
# # print(model(torch.tensor(X)).size)
# X=torch.rand(1,6,18) #Batch x channel x samples
# print(X.shape)
# print(model(X.double()))
