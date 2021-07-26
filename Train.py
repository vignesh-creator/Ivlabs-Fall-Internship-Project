import torch
import loader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import fc_net
import cnn_net
import Resnet
import matplotlib.pyplot as plt


# Hyper parameters
epochs = 5
train_shape = loader.training_data.data.shape[0]
batch_size = loader.batch_size
lr = 2e-3
training_loader = loader.training_loader

#Network
while(1):
    flag = int(input("Click 1 to use FC Net, 2 to use CNN Net, 3 to use ResNet:"))
    if flag == 1:
        net = fc_net.net
        break
    elif flag == 2:
        net = cnn_net.net
        break
    elif flag == 3:
        net = Resnet.net
        break
    else:
        print("Enter Valid Number")


#Optimiser

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

#Training the Network
loss_list=[]
correct=0
total=0     

for epoch in range(epochs):
    for data in training_loader:  
        X, Y = data  
        X = X.to(device)
        Y = Y.to(device)
        net.zero_grad()
        if flag == 1:
            output = net(X.view(-1,784)) 
        else:
            output = net(X)
        label = F.one_hot(Y).type(torch.float32)
        loss = loss_function(output,label) 
        loss_list.append(loss.item())
        loss.backward()  
        optimizer.step()
        for idx, i in enumerate(output):
            if torch.argmax(i) == Y[idx]:
                correct += 1
            total += 1
    print("For iteration:",epoch+1,"Loss is:",loss.item(),"Accuracy is:",(correct/total)*100)


plt.plot(loss_list)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost')
plt.show()
