from argparse import ArgumentParser
import os
import h5py
import torch
from torch.optim import Adam, lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import random
from scipy import stats
import cv2
import datetime
import pandas as pd
import sys
from tqdm import tqdm
import collections

class VQADataset(Dataset):
    def __init__(self, features_dir3, index=None, max_len=8000, feat_dim=4096, scale=1):
        super(VQADataset, self).__init__()
        self.folders = index
        self.features_dir = features_dir3
        self.max_len = 240
        self.feat_dim = feat_dim
        self.scale = scale
    def __len__(self):
        return len(self.folders)

    
    def get_img(self,path):
        
        data = np.zeros((self.max_len, self.feat_dim))
        features = np.load(self.features_dir + path)
        if features.shape[0] > self.max_len:
            features = features[0:self.max_len,:]
        length = features.shape[0]
        data[:length,:] = features
        
        
        label = float(path.split("--")[1][0:-4])/self.scale
        
        name = path.split("_")[0]

        return data,length,label,name
        
        
    def __getitem__(self, idx):
        
        img_data,length,label,name= self.get_img(self.folders[idx])
        
        
        
        
        sample = img_data,length,label,name
        return sample
    
class VQA_padding(Dataset):
    def __init__(self, features_dir3, index=None, max_len=8000, feat_dim=4096, scale=1,padding_shape=None):
        super(VQA_padding, self).__init__()
        self.folders = index
        self.features_dir = features_dir3
        self.max_len = 12000
        self.feat_dim = feat_dim
        self.scale = scale
        self.padding_shape = padding_shape
    def __len__(self):
        return len(self.folders)

    
    def get_img(self,path):
        
       # data = np.zeros((max_len, self.feat_dim))
        features = np.load(self.features_dir + path)
        if features.shape[0] > self.max_len:
            features = features[0:self.max_len,:]
        length = features.shape[0]
        
        
#         for i in range(len(self.padding_shape)-1):
#             if self.padding_shape[i] < length and self.padding_shape[i+1] > length:
#                 data = np.zeros((self.padding_shape[i+1], 4096))
#                 data[:length,:] = features    
#                 features = data
         
        label = int(path.split("--")[1][0])

        name = path.split("_")[0]
        if self.padding_shape:
            p_size = self.padding_shape
            data = np.zeros((p_size, 4096))
            data[:length,:] = features
            features = data
        return features,length,label,name
        
        
    def __getitem__(self, idx):
        
        img_data,length,label,name= self.get_img(self.folders[idx])
        
        
        
        
        sample = img_data,length,label,name
        return sample
def split_and_padding(data_list):
    
    len_dict = collections.defaultdict(list)
    
    for features,length,label,name in tqdm(data_list):
        
        with open("split_padding.txt","a") as f:
            padd = 0
            if length <2000:
                padd = 2000
#             elif length <4000:
#                 padd = 4000
#             elif length <6000:
#                 padd = 6000
#             elif length <8000:
#                 padd = 8000
#             elif length <10000:
#                 padd = 10000
#             else:
#                 padd = 12000
            f.write(name+"\t")
            f.write(str(padd)+"\n")  
    
    return len_dict

class FPN_VQADataset(Dataset):
    def __init__(self, features_dir3, index=None, max_len=8000, feat_dim=4096, scale=1):
        super(FPN_VQADataset, self).__init__()
        self.folders = index
        self.features_dir = features_dir3
        self.max_len = 240
        self.feat_dim = feat_dim
        self.scale = scale
    def __len__(self):
        return len(self.folders)

    
    def get_img(self,path):
        
        data = np.zeros((self.max_len, self.feat_dim))
        data_256 = np.zeros((self.max_len, 256,16,16))
        data_512 = np.zeros((self.max_len, 512,8,8))
        data_1024 = np.zeros((self.max_len, 1024,4,4))
        
        
        
        all_features = np.load(self.features_dir + path,allow_pickle=True)
        
        features = all_features[0].cpu()
        length = features.shape[0]
      
        data[:length,:] = features
      
        down_256 = all_features[1].cpu()
        data_256[:length,:] = down_256
        
        down_512 = all_features[2].cpu()
        data_512[:length,:] = down_512
        
        down_1024 = all_features[3].cpu()
        data_1024[:length,:] = down_1024

        down_list = [data_256,data_512,data_1024]
#         for i in down_list:
#             print("i.shape",i.shape)
        
        label = float(path.split("--")[1][0:-4])/self.scale
        
        name = path.split("_")[0]

        return data,down_list,length,label,name
        
        
    def __getitem__(self, idx):
        
        img_data,down_list,length,label,name= self.get_img(self.folders[idx])
        
        
        
        
        sample = img_data,down_list,length,label,name
        return sample
    

def RGB2Gray(R, G, B):
	res = int(0.299 * R + 0.587 * G + 0.114 * B)
	if res > 255:
		return 255
	else:
		return res
 


def make_dliated(video_data):
   
    length = video_data.shape[0]
    w = video_data.shape[1]
    h = video_data.shape[2]
    
    
    
    dliated_data = np.zeros((length, w,h,1))
    
    for i in range(length):

        gray = cv2.cvtColor(video_data[i,:, :, :],cv2.COLOR_BGR2GRAY)
       # print(type(gray))
        gray = gray[:,:,np.newaxis]
 

        if i == 0:
            dliated_data[i,:,:,:] = gray
      #  print(dliated_data.shape)
            tmp = gray
        else:
            dliate_gray = abs(gray-tmp)

            tmp = gray
            dliated_data[i,:,:,:] = dliate_gray
#     negavg = np.mean(dliated_data[dliated_data < 0.0])
#     print(negavg)
    return dliated_data
                
                
                
		#new_imarray[i, j, 1] = new_imarray[i, j, 0]
		#new_imarray[i, j, 2] = new_imarray[i, j, 0] 
    
class Dliated_VQADataset(Dataset):
    def __init__(self, features_dir3, index=None, max_len=8000, feat_dim=4096, scale=1):
        super(Dliated_VQADataset, self).__init__()
        self.folders = index
        self.features_dir = features_dir3
        self.max_len = 240
        self.feat_dim = feat_dim
        self.scale = scale
        self.dilated_dir = "/cfs/cfs-3cab91f9f/liuzhang/open_datasets/Dliated_k_1200/"
    def __len__(self):
        return len(self.folders)

    
    def get_img(self,path):
        
        data = np.zeros((self.max_len, self.feat_dim))
        D_data = np.zeros((self.max_len, self.feat_dim))
        
        features = np.load(self.features_dir + path)
        if features.shape[0] > self.max_len:
            features = features[0:self.max_len,:]
            
            
        new_path = path.split("_")[0]+"_dliated_"+path.split("_")[2]
        D_features = np.load(self.dilated_dir + new_path)
        
        
        
        
        length = features.shape[0]
        data[:length,:] = features
        D_data[:length,:] = D_features
        
        label = float(path.split("--")[1][0:-4])/self.scale
        
        name = path.split("_")[0]

        return data,D_data,length,label,name
        
        
    def __getitem__(self, idx):
        
        img_data,D_data,length,label,name= self.get_img(self.folders[idx])
        
        
        
        
        sample = img_data,D_data,length,label,name
        return sample
    
    
class FPN_Dliated_VQADataset(Dataset):
    def __init__(self, features_dir3, index=None, max_len=8000, feat_dim=4096, scale=1):
        super(FPN_Dliated_VQADataset, self).__init__()
        self.folders = index
        self.features_dir = features_dir3
        self.max_len = 240
        self.feat_dim = feat_dim
        self.scale = scale
        self.dilated_dir = "/cfs/cfs-3cab91f9f/liuzhang/open_datasets/FPN_D_30_dliated_k_1200/"
    def __len__(self):
        return len(self.folders)

    
    def get_img(self,path):
        
        data = np.zeros((self.max_len, self.feat_dim))
        D_data = np.zeros((self.max_len, self.feat_dim))
        
        features = np.load(self.features_dir + path)
        if features.shape[0] > self.max_len:
            features = features[0:self.max_len,:]
            
            
        new_path = path.split("_")[0]+"_"+path.split("_")[1]+"_"+path.split("_")[2]+"_D"+path.split("_")[3][1:]
        D_features = np.load(self.dilated_dir + new_path)
        
        
        
        
        length = features.shape[0]
        data[:length,:] = features
        D_data[:length,:] = D_features
        
        label = float(path.split("--")[1][0:-4])/self.scale
        
        name = path.split("_")[0]

        return data,D_data,length,label,name
        
        
    def __getitem__(self, idx):
        
        img_data,D_data,length,label,name= self.get_img(self.folders[idx])
        
        
        
        
        sample = img_data,D_data,length,label,name
        return sample
     