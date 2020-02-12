# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from simple_convnet import SimpleConvNet
import pandas as pd
from PIL import Image
from torchvision import transforms
from os.path import isfile, join
from os import listdir

one_hot_key = {
	0:'Black-grass',
	1:'Charlock',
	2:'Cleavers',
	3:'Common Chickweed',
	4:'Common wheat',
	5:'Fat Hen',
	6:'Loose Silky-bent',
	7:'Maize',
	8:'Scentless Mayweed',
	9:'Shepherds Purse',
	10:'Small-flowered Cranesbill',
	11:'Sugar beet'
}

def test():
    
    network = SimpleConvNet(input_dim=(3,128,128), 
                        conv_param = {'filter_num': [32,32], 'filter_size': [3,3], 'pad': [1,1], 'stride': [1,1]},
                        hidden_size=100, output_size=12, weight_init_std=0.01)

    network.load_params("params.pkl")

    data_transform = transforms.Compose([
        transforms.RandomResizedCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    path = join('dataset\plant-seedlings-classification', 'test')

    sample_submission = pd.read_csv('sample_submission.csv')
    submission = sample_submission.copy()
    for i, f in enumerate(sample_submission['file']):
        print(join(path, f))
        image = Image.open(join(path, f))

        #convert image
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = data_transform(image)
        image = np.expand_dims(image, axis=0)
        
        output = network.predict(image)
        output = np.argmax(output, axis=1)
        
        submission['species'][i] = one_hot_key[output[0]]

    submission.to_csv('submission.csv', index=False)

if __name__ == '__main__':
    test()
