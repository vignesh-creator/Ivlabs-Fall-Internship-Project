import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#This is the Entire residual block which has two sub blocks in it
class block(nn.Module):
    def __init__(self,in_channel,intermediate_channel,downsample=None,stride=1):
        super().__init__()
        self.expansion = 4
        self.block1 = nn.Sequential(
                    nn.Conv2d(in_channel,intermediate_channel,3,stride,1).to(device),
                    nn.BatchNorm2d(intermediate_channel).to(device),
                    nn.ReLU()
                )
        self.block2 = nn.Sequential(
                    nn.Conv2d(intermediate_channel,intermediate_channel,3,1,1).to(device),
                    nn.BatchNorm2d(intermediate_channel).to(device),
                    nn.ReLU()
                )
        self.downsample = downsample
        self.stride = stride
    def forward(self,x):
        initial = x
        x = self.block1(x)
        x = self.block2(x)
        if self.downsample is not None:
            initial = self.downsample(initial)
        x += initial
        x = F.relu(x)
        return x


#ResNet is implemented below
class ResNet(nn.Module):
    def __init__(self,img_channel,layers,block,classes):
        super().__init__()
        self.img_channel = img_channel
        self.in_channel = 64 
        self.initial_layer = nn.Sequential(
                                    nn.Conv2d(self.img_channel,self.in_channel,kernel_size=7,stride=2,padding=3).to(device),
                                    nn.BatchNorm2d(self.in_channel).to(device),
                                    nn.MaxPool2d(kernel_size=3,stride=2,padding =1)
                                )
        self.layer_1 = self.create_layer(block,layers[0],64,stride=1)
        self.layer_2 = self.create_layer(block,layers[1],128,stride=2)
        self.layer_3 = self.create_layer(block,layers[2],256,stride=2)
        self.layer_4 = self.create_layer(block,layers[0],512,stride=2)
        
        self.fc_layer = nn.Sequential(
                                        nn.Flatten(),
                                        nn.Linear(512,classes).to(device)
                                        )
        
    def forward(self,x):
        x = self.initial_layer(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.fc_layer(x)
        return torch.sigmoid(x)
    
    def create_layer(self,block,residual_blocks,intermediate_channel,stride):
        identity_downsample = None
        layers = []
        if stride!=1 or self.in_channel!=intermediate_channel:
            identity_downsample = nn.Sequential(
                                nn.Conv2d(self.in_channel,intermediate_channel,1,stride).to(device),
                                nn.BatchNorm2d(intermediate_channel).to(device)
            )
        layers.append(block(self.in_channel,intermediate_channel,identity_downsample,stride))
        self.in_channel = intermediate_channel
        for i in range(1,residual_blocks):
            layers.append(block(self.in_channel,intermediate_channel))
        return nn.Sequential(*layers)

''' 
. While implementing the resnet, we need to pass four parameters into the ResNet Module(check the line 37). The parameters are image 
  channels,layers,block,classes.
. Image channels is basically the channels of images(Example: img_channel for grey scale image = 1 & for RGB image=3)
. Layers is the most important part which represents number of layers are included in blocks.(Example: If one particular block is of 
  size 2,then it has 2*(2 subblock layers) = 4 layers)
. Block is basically the residual block(Line 8)
. Classes is actually the Labels if you are training a Supervised model(Example: if you're making a model to classify cat,dog and pig,
  then number of Classes=3)
. This is the example of implementing the ResNet
. Example : ResNet(1,[1,1,1,1],block,10)

'''

net = ResNet(1,[1,1,1,1],block,10)
