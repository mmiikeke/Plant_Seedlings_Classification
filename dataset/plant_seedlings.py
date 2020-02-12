# coding: utf-8
try:
    import urllib.request
except ImportError:
    raise ImportError('You should use Python 3.x')
import os.path
import gzip
import pickle
import os
from pathlib import Path
import numpy as np
import cv2
from torchvision import transforms
from os.path import isfile, join
from os import listdir
from PIL import Image

dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/plant_seedlings.pkl"

train_num = 60000
test_num = 10000
img_dim = (1, 224, 224)
img_size = 224*224

def _load_data(dataset, name):
    data_transform = transforms.Compose([
        transforms.RandomResizedCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
	
    counter = 0
    num_classes = 0
    path = join('dataset\plant-seedlings-classification', name)
    for f in listdir(path):
        path2 = join(path, f)
        for f2 in listdir(path2):
            print(join(path2, f2))
            image = Image.open(join(path2, f2))

            #convert image
            if image.mode != "RGB":
                image = image.convert("RGB")
            image = data_transform(image)
            image = np.expand_dims(image, axis=0)

            if counter == 0:
                x = image
                y = np.expand_dims(num_classes, axis=0)
            else:
                x = np.append(x, image, axis=0)
                y = np.append(y, num_classes)
            
            counter += 1

        num_classes += 1

    return x, y

def _load_data2(dataset, name):
    data_transform = transforms.Compose([
        transforms.RandomResizedCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    counter = 0
    path = join('dataset\plant-seedlings-classification', name)
    for f in listdir(path):
        print(join(path, f))
        image = Image.open(join(path, f))

        #convert image
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = data_transform(image)
        image = np.expand_dims(image, axis=0)

        if counter == 0:
            x = image		
        else:
            x = np.append(x, image, axis=0)

        counter += 1

    return x

def _convert_numpy():
	
    dataset = {}
    dataset['train_img'], dataset['train_label'] = _load_data(dataset, 'train')
    #dataset['validation_img'] = _load_data2(dataset, 'test')

    return dataset
	
def init_plant_seedlings():
    dataset = _convert_numpy()

    print("Creating pickle file ...")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done!")

def load_plant_seedlings():
    if not os.path.exists(save_file):
        init_plant_seedlings()
    
    print("Reading pickle file ...")
    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)
    print("Done!")
	
    return dataset['train_img'], dataset['train_label']

if __name__ == '__main__':
    init_mnist()
