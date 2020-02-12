from pathlib import Path
import numpy as np
from os import listdir
from os.path import isfile, join
from PIL import Image
from torchvision import transforms
import cv2

def _load_data(name):
    data_transform = transforms.Compose([
        transforms.RandomResizedCrop(128)#,
        #transforms.ToTensor()#,
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
	
    counter = 0
    num_classes = 0
    path = join('plant-seedlings-classification', 'train')
    for f in listdir(path):
        path2 = join(path, f)
        for f2 in listdir(path2):
            print(join(path2, f2))
            image = Image.open(join(path2, f2))

            #convert image
            if image.mode != "RGB":
                image = image.convert("RGB")
            image = data_transform(image)
            image.show()
            break
            counter += 1
        break
        num_classes += 1

    #return x, y
	
	
if __name__ == '__main__':
    _load_data('plant-seedlings-classification/train')