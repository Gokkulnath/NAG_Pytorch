import torch
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from torchvision.datasets.folder import DatasetFolder,ImageFolder

import numpy as np
from glob import glob
from PIL import Image
import pandas as pd
import os,time,gc
from pathlib import Path
from tqdm import tqdm_notebook as tqdm
import datetime,random,string

ngpu=torch.cuda.device_count()
device = torch.device("cuda" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
# transforms
size=224
# Imagenet Stats
vgg_mean = [103.939, 116.779, 123.68]

preprocess=transforms.Compose([transforms.Resize((size,size)),
                               transforms.ToTensor(),
                               transforms.Normalize(vgg_mean,(0.5, 0.5, 0.5))])


print("Using Pytorch Version : {} and Torchvision Version : {}. Using Device {}".format(torch.__version__,torchvision.__version__,device))

dataset_path=r'../data/ILSVRC/'
train_dataset_path=dataset_path+'train'
test_dataset_path=dataset_path+'valid'
print("Dataset root Folder:{}. Train Data Path: {}. Validation Data Path {}".format(dataset_path,train_dataset_path,test_dataset_path))

# Preparation of Labels 
label_dict={}
label_idx={}

with open(f'{dataset_path}/LOC_synset_mapping.txt') as file:
    lines=file.readlines()
    for idx,line in enumerate(lines):
        label,actual =line.strip('\n').split(' ',maxsplit=1)
        label_dict[label]=actual
        label_idx[label]=idx
        
class CustomDataset(Dataset):
    def __init__(self, subset, root_dir, transform=None):
        self.root_dir=root_dir
        self.transform=transform
       
        self.subset=subset
        if self.subset=='train':
            data_dir=os.path.join(self.root_dir,self.subset)
            self.images_fn=glob(f'{data_dir}/*/*')
            self.labels=[Path(fn).parent.name for fn in self.images_fn]
        elif subset =='valid':
            df=pd.read_csv('ILSVRC/LOC_val_solution.csv')
            df['label']=df['PredictionString'].str.split(' ',n=1,expand=True)[0]
            df=df.drop(columns=['PredictionString'])
            self.images_fn='ILSVRC/valid/'+df['ImageId'].values+'.JPEG'
            self.labels=df['label']
        else:
            raise ValueError
        print(f" Number of instances in {self.subset} subset of Dataset: {len(self.images_fn)}")       

    def __getitem__(self,idx):
        fn=self.images_fn[idx]
        label=self.labels[idx]
        image=Image.open(fn)
        if image.getbands()[0] == 'L':
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)    
        return image,label_idx[label]
    
    def __len__(self):
        return len(self.images_fn)
        
# data_train=ImageFolder(root='ILSVRC/train',transform=preprocess)
# class2idx=data_train.class_to_idx
# data_valid=CustomDataset(subset='valid',root_dir=dataset_path,transform=preprocess)

# train_num = len(data_train)
# val_num = len(data_valid)