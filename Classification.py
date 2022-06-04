from glob import glob
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import random

indoor_imgs=[]
outdoor_imgs=[]
    
def gray_imread(indoor , outdoor):
    for path in indoor:
        i_img = cv2.imread(path)
        i_img = cv2.cvtColor(i_img,cv2.COLOR_BGR2GRAY)
        indoor_imgs.append(i_img)
        
    for path in outdoor:        
        o_img = cv2.imread(path)
        o_img = cv2.cvtColor(o_img,cv2.COLOR_BGR2GRAY)
        outdoor_imgs.append(o_img)
           
        
def Train_Test():
    Y = []
    X = []
    for i in range(len(indoor_imgs)):
        Y.append(1)
        X.append(indoor_imgs[i])

    for i in range(len(outdoor_imgs)):
        Y.append(0)
        X.append(outdoor_imgs[i])
    
    random.Random(1).shuffle(X)
    random.Random(1).shuffle(Y)
    
    print(Y)
    
    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.33 , random_state=33)
    y_train = to_categorical(y_train,2)
    y_test = to_categorical(y_test,2)
    return x_train, x_test, y_train, y_test
        
        

        
def alexnet(pretrained_weights = None,input_size = (256,256,1)):
    model = Sequential()
    model.add(Conv2D(input_shape=input_size, filters= 512, kernel_size =(5,5), activation = "relu"))
    model.add(Conv2D(filters= 512, kernel_size =(5,5) ,activation = "relu"))
    model.add(Conv2D(filters= 256, kernel_size =(5,5) , activation = "relu"))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
    
    model.add(Conv2D(filters= 256, kernel_size =(3,3) , activation = "relu" , padding='same'))
    model.add(Conv2D(filters= 256, kernel_size =(3,3) , activation = "relu" , padding='same'))
    model.add(Conv2D(filters= 128, kernel_size =(3,3) , activation = "relu" , padding='same'))
    
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
    
    model.add(Conv2D(filters= 128,kernel_size =(3,3), activation = "relu" , padding='same'))
    model.add(Conv2D(filters= 128 , kernel_size =(3,3), activation = "relu" , padding='same'))
    model.add(Conv2D(filters= 64 , kernel_size =(3,3) , activation = "relu" , padding='same'))
    
    model.add(Flatten())
    model.add(Dense(256 , activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(256 , activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(2 , activation='softmax'))
    
    return model
    
    
def model_fit(x_train,x_test,y_train,y_test):
    model = alexnet()
    checkpoint_filepath = 'your_path'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)
    
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    model.fit(x_train,y_train , epochs=50, validation_data=(x_test,y_test) ,batch_size=2,
              callbacks=[model_checkpoint_callback],shuffle=1)
    model.load_weights(checkpoint_filepath)
    
  
indoor_paths = glob('your_path')  
outdoor_paths = glob('your_path') 
 
gray_imread(indoor_paths, outdoor_paths) 

x_train,x_test,y_train,y_test = Train_Test()
x_train = np.array(x_train).reshape(-1,256,256,1)
x_test = np.array(x_test).reshape(-1,256,256,1)
model_fit(x_train,x_test,y_train,y_test)   

    
    
        