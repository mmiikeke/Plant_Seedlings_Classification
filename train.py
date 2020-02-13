from torch.utils.data import Dataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn as nn
import torch

from torchvision import transforms
from PIL import Image
import math
import copy
from os.path import join
from os import listdir
import os.path
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class PlantSeedlingDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.x = []
        self.y = []
        self.transform = transform
        self.num_classes = 0
        
        i = 0
        
        for f in listdir(root_dir):
            path2 = join(root_dir, f)
            for f2 in listdir(path2):
                print(join(path2, f2))
                self.x.append(join(path2, f2))
                self.y.append(i)
            self.num_classes += 1
            i += 1

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        image = Image.open(self.x[index]).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, self.y[index]

class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            # input shape: (batch_size, 3, 224, 224) and
            # downsampled by a factor of 2^3 = 8 (3 times maxpooling)
            # So features' shape is (batch_size, 28, 28, 512)
            nn.Linear(in_features=28 * 28 * 256, out_features=2048),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=2048, out_features=2048),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=2048, out_features=num_classes)
        )

        # initialize parameters
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2. / n))
                module.bias.data.zero_()
            elif isinstance(module, nn.Linear):
                module.weight.data.normal_(0, 0.01)
                module.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        # flatten
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


CUDA_DEVICES = 0
train_loss_list = []
validation_loss_list = []
train_acc_list = []
validation_acc_list = []

def train():
    data_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    data_set = PlantSeedlingDataset('plant-seedlings-classification/train', data_transform)
    train_set = PlantSeedlingDataset('plant-seedlings-classification/train', data_transform)
    validation_set = PlantSeedlingDataset('plant-seedlings-classification/train', data_transform)
    
    #train test split
    dataset_size = len(data_set)
    validation_split = .2
    split = int(np.floor(validation_split * dataset_size))
    array_all = np.arange(dataset_size)
    random.shuffle(array_all)
    array_validation = array_all[:split]
    array_train = array_all[split:]
    
    train_set.x = [data_set.x[idx] for idx in array_train]
    train_set.y = [data_set.y[idx] for idx in array_train]
    validation_set.x = [data_set.x[idx] for idx in array_validation]
    validation_set.y = [data_set.y[idx] for idx in array_validation]
    
    train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True, num_workers=1)
    validation_loader = DataLoader(dataset=validation_set, batch_size=32, shuffle=True, num_workers=1)


    model = VGG16(num_classes=data_set.num_classes)
    model = model.cuda(CUDA_DEVICES)
    model.train()

    best_model_params = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    num_epochs = 80
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(num_epochs):
        print(f'Epoch: {epoch + 1}/{num_epochs}')
        print('-' * len(f'Epoch: {epoch + 1}/{num_epochs}'))

        training_loss = 0.0
        training_corrects = 0.0
        validation_loss = 0.0
        validation_corrects = 0.0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = Variable(inputs.cuda(CUDA_DEVICES))
            labels = Variable(labels.cuda(CUDA_DEVICES))

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            
            training_loss += loss.data * inputs.size(0)
            training_corrects += torch.sum(preds == labels.data)

        training_loss = float(training_loss) / (dataset_size - split)
        training_acc = float(training_corrects) / (dataset_size - split)

        print(f'Train loss: {training_loss:.4f}\taccuracy: {training_acc:.4f}\n')
        
        for i, (inputs, labels) in enumerate(validation_loader):
            inputs = Variable(inputs.cuda(CUDA_DEVICES))
            labels = Variable(labels.cuda(CUDA_DEVICES))

            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            
            validation_loss += loss.data * inputs.size(0)
            validation_corrects += torch.sum(preds == labels.data)
        
        validation_loss = float(validation_loss) / split
        validation_acc = float(validation_corrects) / split

        print(f'Validation loss: {validation_loss:.4f}\taccuracy: {validation_acc:.4f}\n')
        
        train_acc_list.append(training_acc)
        validation_acc_list.append(validation_acc)
        train_loss_list.append(training_loss)
        validation_loss_list.append(validation_loss)
        
        if training_acc > best_acc:
            best_acc = training_acc
            best_model_params = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_params)
    torch.save(model, f'model-weight_and_bias.pth')

def show_acc_and_loss():
    x = np.arange(len(train_acc_list))

    plt.subplot(2,1,1)
    plt.plot(x, train_acc_list, marker='o', label='train', markevery=2)
    plt.plot(x, validation_acc_list, marker='s', label='validation', markevery=2)
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')

    plt.subplot(2,1,2)
    plt.plot(x, train_loss_list, marker='o', label='train', markevery=2)
    plt.plot(x, validation_loss_list, marker='s', label='validation', markevery=2)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend(loc='upper right')

    plt.show()
	
if __name__ == '__main__':
    train()
