import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchmetrics import Accuracy, Precision, Recall

# Load datasets
from torchvision import datasets
import torchvision.transforms as transforms

train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
dataloader_train = DataLoader(train_data, batch_size=10, shuffle = True)
dataloader_test = DataLoader(test_data, batch_size=10, shuffle = True)

import torch.nn as nn
import torch.optim as optim

# Defining the CNN
class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net,self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten()
        )
        self.classifier = nn.Linear(64*7*7, num_classes)  # Updated the input size to the classifier
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

# Training the CNN
net = Net(num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
for epoch in range(10):
    for images, labels in dataloader_train:
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Testing the CNN
metric_accuracy = Accuracy(task='multiclass', num_classes=10, average=None)
metric_precison = Precision(task='multiclass', num_classes=10, average=None)
metric_recall = Recall(task='multiclass', num_classes=10, average=None)

net.eval()
with torch.no_grad():
    for images, labels in dataloader_test:
        outputs = net(images)
        _, predictions = torch.max(outputs,1)
        metric_accuracy.update(predictions, labels)
        metric_precison.update(predictions, labels)
        metric_recall.update(predictions,labels)

# Torchmetrics Calculation
accuracy = metric_accuracy.compute()
precision = metric_precison.compute()
recall = metric_recall.compute()

print(f"Accuracy : {accuracy}")   
print(f"Precision : {precision}")
print(f"Recall : {recall}")

recall_per_class = {K: recall[v].item() for K, v in test_data.class_to_idx.items()}
print(recall_per_class)
precision_per_class = {K: precision[v].item() for K,v in test_data.class_to_idx.items()}
print(precision_per_class)
        






