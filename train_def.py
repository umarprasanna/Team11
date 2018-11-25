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
import skimage.io as io
import pdb
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
        img = load_img(path + '/images/' + id_, grayscale=True)
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
    

def image_generator(batch_size,train):
    """
    Custom image generator that adds Gaussian noise and generates
    batches of images and masks arrays
    
    """
    mean = 0.0
    variance = np.random.uniform(0.01, 1.0)
    border = 5
    
    path_train = "./"
    
    #list of files
    train_ids = next(os.walk(path_train+"images"))[2]
    
    X = np.zeros((batch_size, 128, 128, 2), dtype=np.float32)
    if train:
        y = np.zeros((batch_size, 128, 128,1), dtype=np.float32)
    
    images,masks = get_data(path_train,train_ids,train=True) # Get arrays  
    
    batch_images = np.zeros((batch_size, 101,101,3))
    batch_masks = np.zeros((batch_size,101,101,1))
    print('Generating new images........')
    
    if batch_size < 4000:
      raise ValueError("You cannot input less than 4000")
    
    for i in range(batch_size):
        if i<4000:
           # choose random index in features
           batch_images[i] = images[i]
           if train:
              batch_masks[i] = masks[i]

           x_image = resize(batch_images[i], (128, 128, 1), mode='constant', preserve_range=True)

           # Create cumsum x
           x_center_mean = x_image[border:-border, border:-border].mean()
           x_csum = (np.float32(x_image)-x_center_mean).cumsum(axis=0)
           x_csum -= x_csum[border:-border, border:-border].mean()
           x_csum /= max(1e-3, x_csum[border:-border, border:-border].std())

           # Load masks
           if train:
              mask = resize(batch_masks[i], (128, 128, 1), mode='constant', preserve_range=True)
              y[i] = mask / 255

           # Save images
           X[i,...,0] = x_image.squeeze() / 255
           X[i, ..., 1] = x_csum.squeeze()

           
            
        else:
           # choose random index in features
           index= np.random.choice(4000,1)[0]
           batch_images[i] = add_random_gaussian_noise(images[index],mean,variance)
           if train:
              batch_masks[i] = masks[index]

           x_image = resize(batch_images[i], (128, 128, 1), mode='constant', preserve_range=True)

           # Create cumsum x
           x_center_mean = x_image[border:-border, border:-border].mean()
           x_csum = (np.float32(x_image)-x_center_mean).cumsum(axis=0)
           x_csum -= x_csum[border:-border, border:-border].mean()
           x_csum /= max(1e-3, x_csum[border:-border, border:-border].std())

           # Load masks
           if train:
              mask = resize(batch_masks[i], (128, 128, 1), mode='constant', preserve_range=True)
              y[i] = mask / 255

           # Save images
           X[i,...,0] = x_image.squeeze() / 255
           X[i, ..., 1] = x_csum.squeeze()

           
    
    print('Done')
    if train:
       return X, y
    else:
       return X
    
    
 
im_width = 128
im_height = 128
border = 5
im_chan = 2 # Number of channels: first is original and second cumsum(axis=0)
n_features = 1 # Number of extra features, like depth

X, y = image_generator(8000,train=True)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15, random_state=42)

save_str = 'model-tgs-salt-var-{0}--noise-doubled-'.format(v) + '-acc-{val_acc:.2f}-epoch-{epoch:02d}.h5'
callbacks = [
    EarlyStopping(patience=7, verbose=1), #was patience=3 for LRNon
    ReduceLROnPlateau(patience=5, verbose=1), #wast patience=3
    ModelCheckpoint(save_str, verbose=1, save_best_only=True, save_weights_only=True, monitor='val_acc')
]

model = UNET.U_Net()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # The mean_iou metrics seens to leak train and test values...

results = model.fit({'img': X_train}, y_train, batch_size=16, epochs=100, callbacks=callbacks, validation_data=({'img': X_valid}, y_valid))
   
