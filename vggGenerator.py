import numpy as np
import keras
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
import pdb

class vggGenerator(keras.utils.Sequence):

    def __init__(self, list_IDs, data_dir, pred_size=4, batch_size=32, dim=(48,48), patch_size= 48, n_channels=1, n_classes=2, shuffle=True):
        self.dim = dim
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.data_dir = data_dir
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.pred_size = pred_size
        self.on_epoch_end()
    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X, y = self.__data_generation__(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation__(self, list_IDs_temp):
        X = np.empty(self.dim + (3, self.batch_size))
        y = np.empty((self.pred_size, self.pred_size) + (self.batch_size,))

        for i, ID in enumerate(list_IDs_temp):
            # Load X
            img = load_img(self.data_dir + '/images/' + ID, grayscale=True)
            x_img = img_to_array(img)
            x_img = np.repeat(resize(x_img, (128, 128, 1), mode='constant', preserve_range=True), 3, 2)
            start_1 = int(np.random.rand() * (x_img.shape[0] - self.patch_size))
            start_2 = int(np.random.rand() * (x_img.shape[1] - self.patch_size))
            end_1 = start_1 + self.patch_size
            end_2 = start_2 + self.patch_size
            ps1 = int((self.patch_size - self.pred_size)/2)
            ps2 = int(self.patch_size - ((self.patch_size - self.pred_size)/2))
            X[:, :, :, i] = x_img[start_1:end_1, start_2:end_2, :]
            mask = img_to_array(load_img(self.data_dir + '/masks/' + ID, grayscale=True))
            mask = resize(mask, (128, 128, 1), mode='constant', preserve_range=True)
            y[:,:,i] = mask[start_1 + ps1:start_1 + ps2, start_2 + ps1:start_2 + ps2, 0]
        return X.reshape((self.batch_size, *self.dim, 3)), y.reshape((self.batch_size, self.pred_size, self.pred_size))

