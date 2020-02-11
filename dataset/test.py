from pathlib import Path
import numpy as np
from os import listdir
from os.path import isfile, join
from PIL import Image
from torchvision import transforms
import cv2

def _load_data(name):
    data_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor()#,
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
	
    x = []
    y = []
    num_classes = 0
    path = 'plant-seedlings-classification/train'
    for f in listdir(path):
        path2 = join(path, f)
        for f2 in listdir(path2):
            print(join(path2, f2))
            image = Image.open(join(path2, f2))
            x.append(data_transform(image))
            y.append(num_classes)

        num_classes += 1

    print(x.shape)
    #im = cv2.imread(x[0])
    #im = np.array(Image.open(x[0]))
    #print(im.shape)
    #im = data_transform(im)
    #print(im.shape)
	
	
if __name__ == '__main__':
    _load_data('plant-seedlings-classification/train')