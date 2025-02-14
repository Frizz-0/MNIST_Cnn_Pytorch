import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import pandas as pd
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import time
import os

# Get the current directory of the file
current_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_directory)


# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5))])
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root = './data', train = True, transform = transform, download = False)
test_dataset = datasets.MNIST(root = './data', train = False, transform = transform, download = False)
train_loader = DataLoader(train_dataset, batch_size = 10, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle = False)


# Lets Create our NN

class MNIST_NN(nn.Module):

    def __init__(self):
        super().__init__()

        #Convolutional Layers
        self.conv1 = nn.Conv2d(1,6,3,1)
        self.conv2 = nn.Conv2d(6,16,3,1)

        #Fully connected Layers
        self.fc1 = nn.Linear(5*5*16, 128)
        self.fc2 = nn.Linear(128 , 64)
        self.fc3 = nn.Linear(64 , 10)

    def forward(self,x):

        #Convolutional Layers and Pooling
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x,2,2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,2,2)

        x = x.view(-1,16*5*5)

        #Fully connected Layers

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return F.log_softmax(x,dim = 1)

    
torch.manual_seed(2)
model = MNIST_NN()

optimizer = optim.Adam(model.parameters(),lr =0.001)
criterion = nn.CrossEntropyLoss()


#Train the data

start_time = time.time()

epochs = 5
losses = []
test_losses = []
train_losses = []
train_crr = []
test_crr = []

for i in range(epochs):

    trn_crr = 0
    tes_crr = 0

    for j, (X_train,y_train) in enumerate(train_loader):

        j += 1

        y_pred = model.forward(X_train)

        loss = criterion(y_pred, y_train)

        losses.append(loss.detach().numpy())

        predicted = torch.max(y_pred.data, 1)[1]
        batch_corr = (predicted == y_train).sum()
        trn_crr += batch_corr


        if j % 600 == 0:

            print(f'Epochs {i} , Batch, {j}  Loss : {loss} ')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_losses.append(loss)
    train_crr.append(trn_crr)

    with torch.no_grad():

        for k, (X_test,y_test) in enumerate(test_loader):

            y_eval = model.forward(X_test)

            predicted = torch.max(y_eval.data, 1)[1] # Adding up correct predictions
            tes_crr += (predicted == y_test).sum() # T=1 F=0 and sum away


    loss = criterion(y_eval, y_test)
    test_losses.append(loss)
    test_crr.append(tes_crr)


current_time = time.time()

total = current_time - start_time
print(f"Training time : {total/60} mins")

#Testing the new model set

# test_data_new = DataLoader(test_dataset, batch_size=10000, shuffle= False)

# with torch.no_grad():
#     correct = 0 
#     for l,(X_test,y_test) in enumerate(test_data_new):
#         y_eval = model(X_test)
#         predicted = torch.max(y_eval,1)[1]
#         correct += (predicted == y_test ).sum()

model.eval()
model.train()

my_mnist_model = MNIST_NN()
torch.save(my_mnist_model.state_dict(), 'state_dict.pth')
