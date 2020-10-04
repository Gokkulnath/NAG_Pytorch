import torch,torchvision
from torch.utils.data import DataLoader,Dataset
from torchvision.datasets.folder import ImageFolder

import numpy as np
from glob import glob
from PIL import Image
import pandas as pd
import os,random
from pathlib import Path
      
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
        self.build_vocab()
        print(f" Number of instances in {self.subset} subset of Dataset: {len(self.images_fn)}")       

    def __getitem__(self,idx):
        fn=self.images_fn[idx]
        label=self.labels[idx]
        image=Image.open(fn)
        if image.getbands()[0] == 'L':
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)    
        return image,self.label_idx[label]
    
    def __len__(self):
        return len(self.images_fn)

    def build_vocab(self):
        # Preparation of Labels 
        self.label_dict={}
        self.label_idx={}

        with open(f'{self.root_dir}/LOC_synset_mapping.txt') as file:
            lines=file.readlines()
            for idx,line in enumerate(lines):
                label,actual =line.strip('\n').split(' ',maxsplit=1)
                self.label_dict[label]=actual
                self.label_idx[label]=idx