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
#just in case we need a backup datasets
from torch.utils import data
#will output the plot right below the cell that produces it


class TGSSaltDataset(data.Dataset):
    #init with the location of the dataset, and the list of file 
    def __init__(self, root_path, file_list,data_type):
        self.root_path = root_path
        self.file_list = file_list
        self.data_type = data_type
        
    #get method - how long is the list
    def __len__(self):
        return len(self.file_list)
      
    #get method - return the seismic image + label for a given index if 
         #training data else returns image only if testing data
    def __getitem__(self,index):
        #if the index is out of bounds, get a random image
        if index not in range(0, len(self.file_list)):
            return self.__getitem__(np.random.randint(0, self.__len__()))
          
        #define a file ID using the index parameter
        file_id = self.file_list[index]
        
        if self.data_type=='train':
           #image folder + path
           image_folder = os.path.join(self.root_path, "images")
           image_path = os.path.join(image_folder, file_id)
        
           #read it, store it in memory as a byte array
           image = np.array(imageio.imread(image_path), dtype=np.uint8)
        
        
           #label folder + path
           mask_folder = os.path.join(self.root_path, "masks")
           mask_path = os.path.join(mask_folder, file_id)
           mask = np.array(imageio.imread(mask_path), dtype=np.uint8)
          
           #return image + label
           return image, mask
        else:
           #image folder + path
           image_folder = os.path.join(self.root_path, "test/images")
           image_path = os.path.join(image_folder, file_id)
        
           #read it, store it in memory as a byte array
           image = np.array(imageio.imread(image_path), dtype=np.uint8)
            
           #return image
           return image
