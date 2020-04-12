# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 11:41:30 2020

@author: SACHUU
"""

import numpy as np # MATRIX OPERATIONS
import pandas as pd # EFFICIENT DATA STRUCTURES
import matplotlib.pyplot as plt # GRAPHING AND VISUALIZATIONS
import math # MATHEMATICAL OPERATIONS
import cv2 # IMAGE PROCESSING - OPENCV
from glob import glob # FILE OPERATIONS
import itertools
import os
import tensorflow as tf

# KERAS AND SKLEARN MODULES
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,CSVLogger

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# GLOBAL VARIABLES
scale = 70
seed = 7


class plant_recognition:
    
    def __init__(self):
        
        print("Recognise Image")
        self.path_to_images = '{}\*\*.jpg'.format(os.path.join(os.getcwd(),'recognition\train'))
        self.images = glob(self.path_to_images)
        self.trainingset = []
        self.traininglabels = []
        self.new_train = []
        self.sets = []
        self.num = len(self.images)
        self.x_train = []
        self.x_test = []
        self.y_train = []
        self.y_test = []
        self.model = Sequential()
        
        
        
        
    def preprocess_images(self):
         
        count = 1
        #READING IMAGES AND RESIZING THEM
        for i in self.images:

            self.traininglabels.append(i.split('\\')[-2])
            count=count+1
        self.traininglabels = pd.DataFrame(self.traininglabels)
    
        
      
        getEx = True
        for i in self.trainingset:
            blurr = cv2.GaussianBlur(i,(5,5),0)
            hsv = cv2.cvtColor(blurr,cv2.COLOR_BGR2HSV)
            #GREEN PARAMETERS
            lower = (25,40,50)
            upper = (75,255,255)
            mask = cv2.inRange(hsv,lower,upper)
            struc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
            mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,struc)
            boolean = mask>0
            new = np.zeros_like(i,np.uint8)
            new[boolean] = i[boolean]
            self.new_train.append(new)
            
            if getEx:
                plt.subplot(2,3,1);plt.imshow(i) # ORIGINAL
                plt.subplot(2,3,2);plt.imshow(blurr) # BLURRED
                plt.subplot(2,3,3);plt.imshow(hsv) # HSV CONVERTED
                plt.subplot(2,3,4);plt.imshow(mask) # MASKED
                plt.subplot(2,3,5);plt.imshow(boolean) # BOOLEAN MASKED
                plt.subplot(2,3,6);plt.imshow(new) # NEW PROCESSED IMAGE
                plt.show()
                getEx = False
        self.new_train = np.asarray(self.new_train)

        for i in range(8):
            plt.subplot(2,4,i+1)
            plt.imshow(self.new_train[i])
            
            
        self.labels = preprocessing.LabelEncoder()
        self.labels.fit(self.traininglabels[0])

        
        self.encodedlabels = self.labels.transform(self.traininglabels[0])
        clearalllabels = np_utils.to_categorical(self.encodedlabels)
        self.classes = clearalllabels.shape[1]
        print('Classes'+str(self.labels.classes_))
        
       
        self.new_train = self.new_train/255
        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(self.new_train,clearalllabels,test_size=0.1,random_state=seed,stratify=clearalllabels)
        

    def create_model(self):


        generator = ImageDataGenerator(rotation_range = 180,zoom_range = 0.1,width_shift_range = 0.1,height_shift_range = 0.1,horizontal_flip = True,vertical_flip = True)
        generator.fit(self.x_train)
        
        np.random.seed(seed)

        
        
        self.model.add(Conv2D(filters=64, kernel_size=(5, 5), input_shape=(scale, scale, 3), activation='relu'))
        self.model.add(BatchNormalization(axis=3))
        self.model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(BatchNormalization(axis=3))
        self.model.add(Dropout(0.1))
        
        self.model.add(Conv2D(filters=128, kernel_size=(5, 5), activation='relu'))
        self.model.add(BatchNormalization(axis=3))
        self.model.add(Conv2D(filters=128, kernel_size=(5, 5), activation='relu'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(BatchNormalization(axis=3))
        self.model.add(Dropout(0.1))
        
        self.model.add(Conv2D(filters=256, kernel_size=(5, 5), activation='relu'))
        self.model.add(BatchNormalization(axis=3))
        self.model.add(Conv2D(filters=256, kernel_size=(5, 5), activation='relu'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(BatchNormalization(axis=3))
        self.model.add(Dropout(0.1))
        
        self.model.add(Flatten())
        
        self.model.add(Dense(256, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.5))
        
        self.model.add(Dense(256, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.5))
        
        self.model.add(Dense(self.classes, activation='softmax'))
        
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        self.model.summary()
        
    def train_model(self):
	
	    preprocess_images()
		create_model()
        
        print(self.x_test[0].shape)
        
        plt.imshow(self.x_test[10])
        history = self.model.fit(self.x_train, self.y_train, epochs=100)

        plt.plot(history.history['loss'])
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()
        
        pred_labels = self.model.predict(self.x_test)
        print(pred_labels.shape)
        acc = self.model.evaluate(self.x_test, self.y_test)
        print("Testing accuracy : {}".format(acc[-1] * 100))
        
        self.save_model()

        
    def save_model(self):

        keras_file = "keras_model.h5"
        self.model.save(keras_file)
        
        # Convert to TensorFlow Lite model.
        converter = tf.lite.TFLiteConverter.from_keras_model_file(keras_file)
        tflite_model = converter.convert()
        open("converted_model.tflite", "wb").write(tflite_model)
		
