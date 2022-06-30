import sys
from osgeo import gdal
import os
import torch
from torch import nn
import torchinfo

from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
import torchvision
from torchvision import models
import matplotlib.pyplot as plt
plt.ion()


class ImData(Dataset):
    def __init__(self,imdir,chipsize=269):
        self.ds_list = []               
        self.chipsize = int(chipsize)
        self.total_chips=0
        
        #Get gdal_datasets
        count=0
        self.start_indices = []
        for r,d,files in os.walk(imdir):
            for f in files:
                if os.path.splitext(f)[1]==".TIF":
                    self.start_indices.append(self.total_chips)
                    self.ds_list.append(gdal.Open(os.path.join(r,f)))
                    count = np.floor(self.ds_list[-1].RasterXSize/self.chipsize)*np.floor(self.ds_list[-1].RasterYSize/self.chipsize)
                    self.total_chips += int(count)
                
    def __len__(self):
        return self.total_chips
    
    def __getitem__(self,i):
        ind_to_use=-1
        im_num=0
        for ds_index,start_ind in enumerate(self.start_indices):
            if i < start_ind:
                ind_to_use = ds_index-1
                break
            im_num=start_ind
        ds = self.ds_list[ind_to_use]
        tile_index=im_num
        
        col_chips = np.floor(ds.RasterXSize/self.chipsize)
        row_chips = np.floor(ds.RasterYSize/self.chipsize)
        col = int((i-tile_index) % np.floor(ds.RasterXSize/self.chipsize))
        row = int(np.floor((i-tile_index) / np.floor(ds.RasterXSize/self.chipsize)))
        
        ar = ds.ReadAsArray(col*self.chipsize,row*self.chipsize,self.chipsize,self.chipsize)
        if ar.max()>0:
            ar = ar / ar.max()
        
        np.transpose(ar,[2,0,1])
        assert(ar.shape[0]<ar.shape[1] and ar.shape[0]<ar.shape[2])
        
        return ar.astype(np.float32)
  
    def write_vrts(self):
        for i in range(self.total_chips):
            ind_to_use=-1
            im_num=0
            for ds_index,start_ind in enumerate(self.start_indices):
                if i < start_ind:
                    ind_to_use = ds_index-1
                    break
                im_num=start_ind
            ds = self.ds_list[ind_to_use]
            tile_index=im_num
            
            col_chips = np.floor(ds.RasterXSize/self.chipsize)
            row_chips = np.floor(ds.RasterYSize/self.chipsize)
            col = int((i-tile_index) % np.floor(ds.RasterXSize/self.chipsize))
            row = int(np.floor((i-tile_index) / np.floor(ds.RasterXSize/self.chipsize)))
            
            ds_sub = gdal.Translate(f'C:/Data/Images/vrts/{i}.vrt',ds,srcWin=[col*self.chipsize,row*self.chipsize,self.chipsize,self.chipsize],format="VRT")
            ds_sub = None

            # ar = ds.ReadAsArray(col*self.chipsize,row*self.chipsize,self.chipsize,self.chipsize)
            # if ar.max()>0:
            #     ar = ar / ar.max()
            
            # np.transpose(ar,[2,0,1])
            # assert(ar.shape[0]<ar.shape[1] and ar.shape[0]<ar.shape[2])
            
        return