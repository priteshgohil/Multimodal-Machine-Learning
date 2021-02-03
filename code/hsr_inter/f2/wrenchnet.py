import numpy
import torch.nn as nn

'''
Input shape : B x 6 x 18
Output shape: B x 16 x 2
'''
class WrenchNet(nn.Module):
    def __init__(self):
        super(WrenchNet, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(108, 256, bias=True), nn.ReLU(inplace=True), nn.Dropout(p = 0.5))
        self.fc2 = nn.Sequential(nn.Linear(256, 1024, bias=True), nn.ReLU(inplace=True), nn.Dropout(p = 0.5))
        self.fc3 = nn.Sequential(nn.Linear(1024, 512, bias=True), nn.ReLU(inplace=True), nn.Dropout(p = 0.5))
        self.fc4 = nn.Sequential(nn.Linear(512, 128, bias=True), nn.ReLU(inplace=True), nn.Dropout(p = 0.5))
        self.fc5 = nn.Linear(128,1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x

# model = WrenchNet().double()
# # print(model(torch.tensor(X)).size)
# X=torch.rand(1,6,18) #Batch x channel x samples
# print(X.shape)
# print(model(X.double()))
