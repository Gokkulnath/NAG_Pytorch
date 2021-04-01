import time,gc
import torch,torchvision
import torch.nn as nn
from torchvision import transforms

from nag.model import model_dict,AdveraryGenerator
from nag.utils import get_device,get_bs
from nag.data import CustomDataset,ImageFolder
from torch.utils.data import DataLoader
from torch import optim
from nag.trainer import fit


total_epochs = 20
lr = 1e-3
# transforms
size=224
# Imagenet Stats
vgg_mean = [0.485, 0.456, 0.406]

preprocess=transforms.Compose([
    transforms.Resize((size,size)),
    transforms.ToTensor(),
])


# Hyperparameters : NAG: Generator
ngf=128
nz= latent_dim=10
e_lim = 10/255
nc=3 # Number of Channels

class Normalize(nn.Module):

    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, input):
        size = input.size()
        x = input.clone()
        for i in range(size[1]):
            x[:, i] = (x[:, i] - self.mean[i]) / self.std[i]

        return x


if __name__ == "__main__":
    device = 'cuda:0'
    print("Using Pytorch Version : {} and Torchvision Version : {}. Using Device {}".format(torch.__version__,torchvision.__version__,device))

    # Setting up the Target Model 
    arch='vgg19'
    print(f"Training Generator for Arch {arch}")
    model= nn.Sequential(
        Normalize(vgg_mean,[0.229, 0.224, 0.225]),
        model_dict[arch](pretrained=True),
    )
    
    # Setting up the NAG : Generator
    G_adversary=AdveraryGenerator(nz,e_lim).to(device)
    optimizer = optim.Adam(G_adversary.parameters(), lr=lr)

    
    # Setting up Dataloaders
    dataset_path=r'/home/DiskA/Dataset/ILSVRC2012_img_val_new'
    data_train=ImageFolder(root=dataset_path+'/train',transform=preprocess)
    # data_train.samples = data_train.samples[:10]
    class2idx=data_train.class_to_idx
    # data_valid=CustomDataset(subset='valid',root_dir=dataset_path,transform=preprocess)
    data_valid=ImageFolder(root=dataset_path+'/val',transform=preprocess)
    train_num = len(data_train)
    val_num = len(data_valid)
    
    bs = get_bs(arch)
    train_dl=DataLoader(data_train,batch_size=bs,shuffle=True,num_workers=4,pin_memory=True,drop_last=True)
    val_dl=DataLoader(data_valid,batch_size=bs,shuffle=True,num_workers=4,pin_memory=True,drop_last=True)
    dls = [train_dl,val_dl]

    fit(total_epochs,model,dls,optimizer,G_adversary,device)
    
    

    
