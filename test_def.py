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
from UNET import U_Net
import skimage.io as io
import skimage.transform as transform
import pdb
#just in case we need a backup datasets
from torch.utils import data
from functions import *

model = U_Net()
model.load_weights('model-tgs-salt-var-0--noise-doubled--acc-0.99-epoch-99.h5')
print('Reading and creating test input')


im_width = 128
im_height = 128
border = 5
im_chan = 2 # Number of channels: first is original and second cumsum(axis=0)
n_features = 1 # Number of extra features, like depth

path_train = "./train/"
path_test = "./"

train_ids = next(os.walk(path_train+"images"))[2][0:3400]
val_ids = next(os.walk(path_train+"images"))[2][3400:4000]

print("Getting training data")
X_t,y_t = get_data(path_train,train_ids,train=True)
print("Getting validation data")
X_v,y_v = get_data(path_train,val_ids,train=True)

print("Converting training and validation data into UNet input format")
X_tarr,y_tarr = array_to_input(X_t,y_t,train=True) 
X_varr,y_varr = array_to_input(X_v,y_v,train=True)

preds_val = model.predict({'img': X_varr}, verbose=1)
thres = np.linspace(0.25, 0.75, 20)
thres_iou = [iou_metric_batch(y_varr, np.int32(preds_val > t)) for t in tqdm(thres)]

best_thres = thres[np.argmax(thres_iou)]
best_thres, max(thres_iou)
path_test = "./"
test_ids = next(os.walk(path_test+"test/images"))[2]


X_te = get_data(path_test,test_ids,train=False)  
y_te = np.zeros((18000, 128, 128,1), dtype=np.float32)
X_test = array_to_input(X_te,y_te,train=False)

print('Predicting for test data')
preds_test = model.predict(X_test,verbose=1)
# Create list of upsampled test masks
preds_test_upsampled = []
for i in range(len(preds_test)):
    preds_test_upsampled.append(resize(np.squeeze(preds_test[i]),(101, 101), mode='constant', preserve_range=True))

pred_dict = {id_[:-4]:RLenc(np.round(preds_test_upsampled[i] > best_thres)) for i,id_ in tqdm(enumerate(test_ids))}

sub = pd.DataFrame.from_dict(pred_dict,orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.to_csv('submission.csv')
files.download('submission.csv')




























