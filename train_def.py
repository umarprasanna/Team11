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
import UNET
#just in case we need a backup datasets
from torch.utils import data

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

def add_random_gaussian_noise(image,mean,variance):
    """
    Generate noise to a given Image based on required noise type
    From: http://www.xiaoliangbai.com/2016/09/09/more-on-image-noise-generation
    
    Input parameters:
        image: ndarray (input image data. It will be converted to float)
       
    """
    row,col,ch= image.shape       
    mu = mean
    var = variance
    sigma = var**0.5
    gauss = np.array(image.shape)
    gauss = np.random.normal(mu,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss
    return noisy.astype('uint8')

def generator(train_ids):
  """
  Generator to return inputs for UNet
  
  """
  # Get and resize train images and masks
  X = np.zeros((len(train_ids), im_height, im_width, im_chan), dtype=np.float32)
  y = np.zeros((len(train_ids), im_height, im_width, 1), dtype=np.float32)
  X_feat = np.zeros((len(train_ids), n_features), dtype=np.float32)
  print('Getting and resizing train images and masks ... ')
  sys.stdout.flush()
  for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
      path = train_path

      # Load X
      img = load_img(path + '/images/' + id_, color_mode='grayscale')

      x_img = img_to_array(img)
      x_img = resize(x_img, (128, 128, 1), mode='constant', preserve_range=True)

      # Create cumsum x
      x_center_mean = x_img[border:-border, border:-border].mean()
      x_csum = (np.float32(x_img)-x_center_mean).cumsum(axis=0)
      x_csum -= x_csum[border:-border, border:-border].mean()
      x_csum /= max(1e-3, x_csum[border:-border, border:-border].std())

      # Load Y
      mask = img_to_array(load_img(path + '/masks/' + id_[0:10] + '.png', color_mode='grayscale'))
      mask = resize(mask, (128, 128, 1), mode='constant', preserve_range=True)

      # Save images
      X[n, ..., 0] = x_img.squeeze() / 255
      X[n, ..., 1] = x_csum.squeeze()
      y[n] = mask / 255
  return X,y

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


train_ids = next(os.walk(train_path+"images"))[2]

im_width = 128
im_height = 128
border = 5
im_chan = 2 # Number of channels: first is original and second cumsum(axis=0)
n_features = 1 # Number of extra features, like depth
#path_train = '../input/train/'
#path_test = '../input/test/'

add_train_ids = add_samples(dataset,train_ids,count=4000,mean=100.0,variance=10.0)
X,y = generator(add_train_ids)

from sklearn.model_selection import train_test_split

X_train, X_valid, X_feat_train, X_feat_valid, y_train, y_valid = train_test_split(X, X_feat, y, test_size=0.15, random_state=20511311)

callbacks = [
    #EarlyStopping(patience=5, verbose=1),
    ReduceLROnPlateau(patience=3, verbose=1),
    ModelCheckpoint('model-tgs-salt-1.h5', verbose=1, save_best_only=True, save_weights_only=True)
]

model = UNET.U_Net()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # The mean_iou metrics seens to leak train and test values...

results = model.fit({'img': X_train, 'feat': X_feat_train}, y_train, batch_size=16, epochs=50, callbacks=callbacks, validation_data=({'img': X_valid, 'feat': X_feat_valid}, y_valid))
