import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 36, 5)
        self.bn3 = nn.BatchNorm2d(36)
        self.conv4 = nn.Conv2d(36, 36, 9)
        self.bn4 = nn.BatchNorm2d(36)
        self.conv5 = nn.Conv2d(36, 50, 7)
        self.bn5 = nn.BatchNorm2d(50)
        self.conv6 = nn.Conv2d(50, 50, 5)
        self.bn6 = nn.BatchNorm2d(50)
        self.fc1 = nn.Linear(50*23*6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):

        # print(x.shape)
        x = (F.relu(self.conv1(x)))
        #print(x.shape)
        x = self.bn1(x)
        x = (F.relu(self.conv2(x)))
        x = self.bn2(x)
        x = (F.relu(self.conv3(x)))
        x = self.bn3(x)
        x = (F.relu(self.conv4(x)))
        x = self.bn4(x)
        x = self.pool(F.relu(self.conv5(x)))
        x = self.bn5(x)
        x = self.pool(F.relu(self.conv6(x)))
        x = self.bn6(x)
        # print(x.shape)
        x = x.view(-1, 50*23*6)
        #print(x.shape)
        x = F.relu(self.fc1(x))
        #print(x.shape)
        x = F.relu(self.fc2(x))
        #print(x.shape)
        x = self.fc3(x)
        #print(x.shape)
        return x