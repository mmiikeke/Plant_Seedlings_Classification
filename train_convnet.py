# coding: utf-8
import sys, os
sys.path.append(os.pardir)  #設置文件父目錄
import numpy as np
import matplotlib.pyplot as plt
from dataset.plant_seedlings import load_plant_seedlings
from simple_convnet import SimpleConvNet
from common.trainer import Trainer
from sklearn.model_selection import train_test_split

#讀取數據
x_data, t_data = load_plant_seedlings()
#減少數據量及運行時間
#x_train, t_train = x_train[:5000], t_train[:5000]
#x_test, t_test = x_test[:1000], t_test[:1000]

x_train, x_test, t_train, t_test = train_test_split(x_data, t_data, test_size=0.2)

print(x_data.shape)
print(x_train.shape)
print(x_test.shape)

max_epochs = 40

network = SimpleConvNet(input_dim=(3,128,128), 
                        conv_param = {'filter_num': [32,32], 'filter_size': [3,3], 'pad': [1,1], 'stride': [1,1]},
                        hidden_size=100, output_size=12, weight_init_std=0.01)
                        
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=200)
trainer.train()

#保存參數
network.save_params("params.pkl")
print("Saved Network Parameters!")

#繪圖
markers = {'train': 'o', 'validation': 's'}
x = np.arange(len(trainer.train_acc_list))

plt.subplot(2,1,1)
plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
plt.plot(x, trainer.test_acc_list, marker='s', label='validation', markevery=2)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')

plt.subplot(2,1,2)
plt.plot(x, trainer.train_loss_list, marker='o', label='train', markevery=2)
plt.plot(x, trainer.test_loss_list, marker='s', label='validation', markevery=2)
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend(loc='upper right')

plt.show()
