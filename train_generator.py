import time,gc
import torch,torchvision
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
vgg_mean = [103.939, 116.779, 123.68]

preprocess=transforms.Compose([transforms.Resize((size,size)),
                               transforms.ToTensor(),
                               transforms.Normalize(vgg_mean,(0.5, 0.5, 0.5))])


# Hyperparameters : NAG: Generator
ngf=128
nz= latent_dim=10
e_lim = 10
nc=3 # Number of Channels




if __name__ == "__main__":
    device = get_device()
    print("Using Pytorch Version : {} and Torchvision Version : {}. Using Device {}".format(torch.__version__,torchvision.__version__,device))

    # Setting up the Target Model 
    arch='resnet50'
    print(f"Training Generator for Arch {arch}")
    model= model_dict[arch](pretrained=True)
    
    # Setting up the NAG : Generator
    G_adversary=AdveraryGenerator(nz,e_lim).to(device)
    optimizer = optim.Adam(G_adversary.parameters(), lr=lr)

    
    # Setting up Dataloaders
    dataset_path=r'ILSVRC'
    data_train=ImageFolder(root='ILSVRC/train',transform=preprocess)
    class2idx=data_train.class_to_idx
    data_valid=CustomDataset(subset='valid',root_dir=dataset_path,transform=preprocess)
    train_num = len(data_train)
    val_num = len(data_valid)
    
    bs = get_bs(arch)
    train_dl=DataLoader(data_train,batch_size=bs,shuffle=True,num_workers=4,pin_memory=True,drop_last=True)
    val_dl=DataLoader(data_valid,batch_size=bs,shuffle=True,num_workers=4,pin_memory=True,drop_last=True)
    dls = [train_dl,val_dl]

    fit(total_epochs,model,dls,optimizer,G_adversary,device)
    
    

    
