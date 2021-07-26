import torch
from torchvision import transforms,datasets,models
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 128


#loading the dataset
training_data = datasets.MNIST(root="data", train=True, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,), (1.0,))
                                  ]))

validation_data = datasets.MNIST(root="data", train=False, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,), (1.0,))
                                  ]))

training_loader = DataLoader(training_data,batch_size=batch_size, shuffle=False,pin_memory=True)
validation_loader = DataLoader(validation_data,batch_size=16,shuffle=False,pin_memory=True)
