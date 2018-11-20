import os
#matrix math
import numpy as np
#read/write image data
import imageio
#visualize data
import matplotlib.pyplot as plt
#data preprocessing 
import pandas as pd
#deep learning
import torch
import TGSSaltDataset as tg
from keras.applications import vgg16 as vgg16
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
#just in case we need a backup datasets
import tensorflow as tf
from torch.utils import data
import pdb
#visualization
def plot2x2Array(image, mask):
    #invoke matplotlib!
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(image)
    axarr[1].imshow(mask)
    axarr[0].grid()
    axarr[1].grid()
    axarr[0].set_title('Image')
    axarr[1].set_title('Mask')

#decoding
def rleToMask(rleString,height,width):
    #width heigh
    rows,cols = height,width
    try:
        #get numbers
        rleNumbers = [int(numstring) for numstring in rleString.split(' ')]
        #get pairs
        rlePairs = np.array(rleNumbers).reshape(-1,2)
        #create an image
        img = np.zeros(rows*cols,dtype=np.uint8)
        #for each pair
        for index,length in rlePairs:
            #get the pixel value 
            index -= 1
            img[index:index+length] = 255
        
        
        #reshape
        img = img.reshape(cols,rows)
        img = img.T
    
    #else return empty image
    except:
        img = np.zeros((cols,rows))
    
    return img

def add_samples(dataset,train_ids,count,mean,variance):
    """
    Creates maximum of 4000 images with noise added. Returns a list of updated train ids inclusive of added noisy images and clean masks
    dataset: dataset class generated object
    count: Number of samples
    mean: Mean of Gaussian noise
    variance: Variance of Gaussian noise
    """

    #Creating samples by adding noise
    list_valid_count = 4000

    if count > list_valid_count:
        raise ValueError("You cannot input more than 4000")

    ids_list = np.random.choice(train_ids,count,replace=False)
    img_count = len(ids_list)
    for n in range(img_count):
        im, mask = dataset[n]
        transformed_image = add_random_gaussian_noise(im,mean,variance)
        save_img(train_path + '/images/'+ train_ids[n].strip('.png')+ '_t'+str(mean)+'_'+str(variance)+ '.png',transformed_image)

    new_train_ids = next(os.walk(train_path+"images"))[2]
    return new_train_ids

train_mask = pd.read_csv('train.csv')
#depth data
depth = pd.read_csv('depths.csv')
#training path
train_path = "./train/"

#list of files
file_list = list(train_mask['id'].values)
#define our dataset using our class
dataset = tg.TGSSaltDataset(train_path, file_list)

import sys
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.transform import resize
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from vggGenerator import vggGenerator

train_ids = next(os.walk(train_path+"images"))[2]
test_ids = train_ids[int(len(train_ids) * .85):len(train_ids)]
train_ids = train_ids[0:int(len(train_ids) * .85)] 

#Add additional samples
new_train_ids = add_samples(dataset,train_ids,count=4000,mean=10.0,variance=0.1)

im_width = 128
im_height = 128
border = 5
im_chan = 3 # Number of channels: first is original and second cumsum(axis=0)
n_features = 1 # Number of extra features, like depth
#path_train = '../input/train/'
#path_test = '../input/test/'
trainG = vggGenerator(train_ids, train_path)
testG = vggGenerator(test_ids, train_path)

callbacks = [
    EarlyStopping(patience=5, verbose=1),
    ReduceLROnPlateau(patience=3, verbose=1),
    ModelCheckpoint('model-tgs-salt-1.h5', verbose=1, save_best_only=True, save_weights_only=True)
]

model = vgg16.VGG16(False, input_shape=(48, 48, 3))
model = Sequential(model.layers)
for layer in model.layers:
         layer.trainable= False
model.add(Flatten())
model.add(Dense(4048, activation='relu'))
#model.add(Activation('softmax'))
model.add(Dense(1024, activation='relu'))
#model.add(Activation('softmax'))
model.add(Dense(1024, activation='relu'))
#model.add(Activation('relu'))
model.add(Dense(512, activation='relu'))
#model.add(Activation('softmax'))
model.add(Dense(1))
model.add(Activation('relu'))
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # The mean_iou metrics seens to leak train and test values...
#results = model.fit({'input_1': X_train, 'feat': X_feat_train}, y_train, batch_size=16, epochs=50, callbacks=callbacks, validation_data=({'img': X_valid, 'feat': X_feat_valid}, y_valid))
#NEED TO REDUCE y TO SMALLER region

results = model.fit_generator(trainG, validation_data=testG, epochs=50, callbacks=callbacks)
