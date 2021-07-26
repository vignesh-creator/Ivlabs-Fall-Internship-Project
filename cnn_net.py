import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
                            nn.Conv2d(1,8,4,2,1), #14
                            nn.ReLU(),
                            nn.Conv2d(8,16,4,2,1),  #7
                            nn.ReLU(),
                            nn.Conv2d(16,32,5,2,1),  #3
                            nn.ReLU(),
                            nn.Conv2d(32,64,5,2,1),  #1
                            nn.ReLU(),
        )
        self.fc=nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(64,10),
        )
    def forward(self, x):
        x=self.conv(x)
        x=self.fc(x)
        return torch.sigmoid(x)

net = Net().to(device)