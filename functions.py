import sys
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img, save_img
from skimage.transform import resize
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from sklearn.model_selection import train_test_split
import os
# math
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
import skimage.io as io
import skimage.transform as transform
import pdb
#just in case we need a backup datasets
from torch.utils import data



def get_data(path,ids,train=True):
    im_width = 101
    im_height = 101
    im_chan = 3
    X = np.zeros((len(ids), im_height, im_width, im_chan), dtype=np.float32)
    if train:
        y = np.zeros((len(ids), im_height, im_width,1), dtype=np.float32)
    print('Getting images ... ')
    for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        # Load images
        if train:
           img = load_img(path + '/images/' + id_, grayscale=True)
           
        else:
           img = load_img(path + 'test/images/' + id_, grayscale=True) #Change path for your system

        x_img = img_to_array(img)
        # Load masks
        if train:
            mask = img_to_array(load_img(path + '/masks/' + id_, grayscale=True))

        # Save images
        X[n] = x_img
        
        if train:
            y[n] = mask
    if train:
        return X, y
    else:
        return X

def array_to_input(X_,y_,train):
  
    border=5
    
    X = np.zeros((len(X_), 128, 128, 2), dtype=np.float32)
    if train:
        y = np.zeros((len(y_), 128, 128,1), dtype=np.float32)
        
    for i in range(len(X_)):
        if train:  # Load masks
           mask = y_[i]
           mask_ = resize(mask, (128, 128, 1), mode='constant', preserve_range=True)
           y[i] = mask_ / 255

        x_image = resize(X_[i], (128, 128, 1), mode='constant', preserve_range=True)

        # Create cumsum x
        x_center_mean = x_image[border:-border, border:-border].mean()
        x_csum = (np.float32(x_image)-x_center_mean).cumsum(axis=0)
        x_csum -= x_csum[border:-border, border:-border].mean()
        x_csum /= max(1e-3, x_csum[border:-border, border:-border].std())  

        # Save images
        X[i,...,0] = x_image.squeeze() / 255
        X[i, ..., 1] = x_csum.squeeze()
        
    if train:
       return X, y
    else:
       return X    
  
      
def add_random_gaussian_noise(image,mean,variance):
    """
    Generate noise to a given Image based on required noise type
    From: http://www.xiaoliangbai.com/2016/09/09/more-on-image-noise-generation
    
    Input parameters:
        image: ndarray (input image data)
       
    """
    row,col,ch= image.shape       
    mu = mean
    var = variance
    sigma = var**0.5
    gauss = np.array(image.shape)
    gauss = np.random.normal(mu,sigma,(row,col,ch))*255
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss
    noisy_img_clipped = np.clip(noisy, 0, 255)
    return noisy_img_clipped

  
def horizontal_flip(image):
    """Flip image left-right"""
    image = cv2.flip(image, 1)
    return image

  
def vertical_flip(image): 
    """Flip image top-bottom"""
    image = cv2.flip(image, 0)
    return image

def image_augmentor(X_,y_,augment):
    mean = 0.0
    variance = 0.01
    border = 5
    
    batch_images = np.zeros((len(X_), 101,101,3))
    batch_masks = np.zeros((len(y_),101,101,1))
        
    for i in range(len(X_)):
        # choose random index in features
        if augment == "noise":
           batch_images[i] = add_random_gaussian_noise(X_[i],mean,variance)
           batch_masks[i] = y_[i]
        elif augment == "lr":
           batch_images[i] = horizontal_flip(X_[i])
           batch_masks[i] = resize(horizontal_flip(y_[i]), (101, 101, 1), mode='constant', preserve_range=True)
        else:
           batch_images[i] = vertical_flip(X_[i])
           batch_masks[i] = resize(vertical_flip(y_[i]), (101, 101, 1), mode='constant', preserve_range=True)
                
    X,y = array_to_input(batch_images,batch_masks,train=True)
    return X,y
    
           

def iou_metric(y_true_in, y_pred_in, print_table=False):
    # src: https://www.kaggle.com/aglotero/another-iou-metric
    labels = y_true_in
    y_pred = y_pred_in
    
    true_objects = 2
    pred_objects = 2

    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins = true_objects)[0]
    area_pred = np.histogram(y_pred, bins = pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union
    
    
    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn
  
# Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)
    
    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)
  
def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.mean(metric)
  

    
def plot2x2Array(image, mask):
    #invoke matplotlib!
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(image)
    axarr[1].imshow(mask)
    axarr[0].grid()
    axarr[1].grid()
    axarr[0].set_title('Image')
    axarr[1].set_title('Mask')  
      
def RLenc(img, order='F', format=True):
    """
    # Source https://www.kaggle.com/bguberfain/unet-with-depth
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e.
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  ## list of run lengths
    r = 0  ## the current run length
    pos = 1  ## count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs

def plot_sample(X, y, preds, binary_preds, ix=None):
    if ix is None:
        ix = random.randint(0, len(X))

    has_mask = y[ix].max() > 0

    fig, ax = plt.subplots(1, 4, figsize=(20, 10))
    ax[0].imshow(X[ix, ..., 0])
    if has_mask:
        ax[0].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[0].set_title('Seismic')

    ax[1].imshow(y[ix].squeeze())
    ax[1].set_title('Salt')

    ax[2].imshow(preds[ix].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[2].contour(y[ix].squeeze(), colors='k', levels=[0.8])
    ax[2].set_title('Salt Predicted')
    
    ax[3].imshow(binary_preds[ix].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[3].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[3].set_title('Salt Predicted binary');

   
