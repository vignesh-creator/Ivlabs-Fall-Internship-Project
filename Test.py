import Train
import torch
import loader
import torch.nn.functional as F
import numpy as np
from torchvision import datasets
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
flag = Train.flag
net = Train.net

validation_loader = loader.validation_loader

correct = 0
total = 0
with torch.no_grad():
    for data in validation_loader:
        X, Y = data
        X = X.to(device)
        Y = Y.to(device)
        output = Train.net(X)
        if flag == 1:
            output = net(X.view(-1,784)) 
        else:
            output = net(X)
        label = F.one_hot(Y).type(torch.float32)
        loss = Train.loss_function(output,label) 
        for idx, i in enumerate(output):
            if torch.argmax(i) == Y[idx]:
                correct += 1
            total += 1

print("Test Loss is:",loss.item(), ",Accuracy: ", (correct/total)*100)

