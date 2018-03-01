import pandas as pd
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
labels_df=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
#print(labels_df.head())
data_dir='F:/deep learning challenge/train_/'
IMG_SIZE=128
LR=1e-3
def read(im):
    img=(cv2.imread(im,cv2.IMREAD_COLOR))
    img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
    return img
print(labels_df['image_name'].values)
test_data=[]
for i in labels_df['image_name'].values:
    test_data.append(read(data_dir+i))
np.save('test_data.npy',test_data)
train_data=np.load('Train_data.npy')
x_train=np.array(train_data,np.float32)/255
#print(x_train.shape)
Class=(labels_df['detected'].tolist())
#print(Class[10])
Y_train = {k:v+1 for v,k in enumerate(set(Class))}
y_train = [Y_train[k] for k in Class]
test_img=np.load('test_data.npy')
x_test = np.array(test_img, np.float32) / 255.

##import tensorflow as tf
##import tflearn
##from tflearn.layers.conv import conv_2d,max_pool_2d
##from tflearn.layers.core import input_data,dropout,fully_connected
##from tflearn.layers.estimator import regression
##
##
##convnet=input_data(shape=[None,IMG_SIZE,IMG_SIZE,3],name='input')
##
##convnet=conv_2d(convnet,32,2,activation='relu')
##convnet=max_pool_2d(convnet,2)
##
##convnet=conv_2d(convnet,64,2,activation='relu')
##convnet=max_pool_2d(convnet,2)
##
##convnet=conv_2d(convnet,32,2,activation='relu')
##convnet=max_pool_2d(convnet,2)
##
##convnet=conv_2d(convnet,64,2,activation='relu')
##convnet=max_pool_2d(convnet,2)
##
##
##convnet=conv_2d(convnet,32,2,activation='relu')
##convnet=max_pool_2d(convnet,2)
##
##convnet=conv_2d(convnet,64,2,activation='relu')
##convnet=max_pool_2d(convnet,2)
##
##convnet=fully_connected(convnet,1024,activation='relu')
##convnet=dropout(convnet,0.8)
##
##
##
##convnet=fully_connected(convnet,64,activation='softmax')
##convnet=regression(convnet,optimizer='adam',learning_rate=LR,loss='categorical_crossentropy')
##
##model=tflearn.DNN(convnet,tensorboard_dir='log')
##model.fit(x_train,y_train,n_epoch=10,
##          snapshot_step=500,show_metric=True)
MODEL_NAME='dl2-{}-{}.model'.format(LR,'keras-basic')
#print((test.row_id))

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Convolution2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.models import load_model
y_train = to_categorical(y_train)

model = Sequential()

model.add(Convolution2D(32, (3,3), activation='relu', padding='same',input_shape = (128,128,3)))
model.add(Convolution2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(64, (3,3), activation='relu', padding='same'))
model.add(Convolution2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Convolution2D(128, (3,3), activation='relu', padding='same'))
model.add(Convolution2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(y_train.shape[1], activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
early_stops = EarlyStopping(patience=3, monitor='val_acc')
model=load_model(MODEL_NAME)
print ('Model loaded!')
model.fit(x_train, y_train, batch_size=100, epochs=20, validation_split=0.3)
model.save(MODEL_NAME)
predictions = model.predict(x_test)

predictions = np.argmax(predictions, axis=1)
rev_y = {v:k for k,v in Y_train.items()}
pred_labels = [rev_y[k] for k in predictions]
sub = pd.DataFrame({'row_id':test.row_id, 'detected':pred_labels})
sub = sub[['row_id', 'detected']]
filename = 'submit.csv'
sub.to_csv(filename, index=False)
