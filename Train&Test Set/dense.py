#!/usr/bin/env python

import keras
import csv
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,Activation,Convolution2D,MaxPooling2D,Flatten,Conv2D
from keras.utils import np_utils
from keras.optimizers import Adam,SGD,Adadelta,RMSprop,Adamax,Adagrad,Nadam
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()

X=np.loadtxt(open('./test/Au_XX_R_TestSet.csv','rb'),delimiter=',',skiprows=1,usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16))
Y=np.loadtxt(open('./test/Au_XX_R_TestSet.csv','rb'),delimiter=',',skiprows=1,usecols=(17))
X = np.array(X)
Y = np.array(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.01, random_state= None)

#Y = np.loadtxt(open('label.txt','rb'),delimiter=',',skiprows=0)

# X_train=np.loadtxt(open('./test/Au_XX_R_TrainSet.csv','rb'),delimiter=',',skiprows=1,usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16))
# Y_train=np.loadtxt(open('Au_XX_R_TrainSet.csv','rb'),delimiter=',',skiprows=1,usecols=(17))
# X_train = np.array(X_train)
# Y_train = np.array(Y_train)

# X_test = np.loadtxt(open('Au_XX_R_TestSet.csv','rb'),delimiter=',',skiprows=1,usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16))
# Y_test = np.loadtxt(open('Au_XX_R_TestSet.csv','rb'),delimiter=',',skiprows=1,usecols=(17))
# X_test = np.array(X_test)
# Y_test = np.array(Y_test)



model = Sequential()
#model.add(Dense(256,input_dim=3))
#model.add(Activation('relu'))
model.add(Dense(128,input_dim=17))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(1))
#model.add(Activation('softmax'))

adam = Adam(lr=0.001)
sgd=SGD(lr=0.01)
adadelta=Adadelta(lr=1.0,rho=0.95,epsilon=1e-06)
rms=RMSprop(lr=0.1)
adagrad =Adagrad(lr=0.01,epsilon=1e-06)
nadam=Nadam(lr=0.002,beta_1=0.9,beta_2=0.999,epsilon=1e-08,schedule_decay=0.004)
adamax=Adamax(lr=0.002,beta_1=0.9,beta_2=0.999,epsilon=1e-08)

model.compile(optimizer=adamax,loss='mse',metrics=['accuracy'])
history = LossHistory()
model.summary()
model.fit(X_train,Y_train,validation_data=(X_test,Y_test),epochs=2000,batch_size=1024,verbose=1,callbacks=[history])
#score=model.evaluate(X_test,Y_test,verbose=1)
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
history.loss_plot('epoch')
#model.save('area_model.h5')
