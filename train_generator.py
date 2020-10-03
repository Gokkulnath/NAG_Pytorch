import time,gc
from nag.model import model_dict
from nag.utils import get_device,get_bs
from nag.data import CustomDataset,ImageFolder
from torch.utils.data import DataLoader
total_epochs = 20
lr = 1e-3
# Setting up Dataloaders
data_train=ImageFolder(root='ILSVRC/train',transform=preprocess)
class2idx=data_train.class_to_idx
data_valid=CustomDataset(subset='valid',root_dir=dataset_path,transform=preprocess)
train_num = len(data_train)
val_num = len(data_valid)
train_dl=DataLoader(data_train,batch_size=bs,shuffle=True,num_workers=4,pin_memory=True,drop_last=True)
val_dl=DataLoader(data_valid,batch_size=bs,shuffle=True,num_workers=4,pin_memory=True,drop_last=True)
dls = [train_dl,val_dl]


device = get_device()
adversarygen = adversarygen=AdveraryGenerator(e_lim).to(device)
optimizer = optim.Adam(adversarygen.parameters(), lr=lr)

print(f"Elsasped Time {time.time()-start} Seconds")

if __name__ == "__main__":
    arch='resnet50'
    start= time.time()
    print(f"Training Generator for Arch {arch}")
    model= model_dict[arch](pretrained=True)
    bs = get_bs(arch)
    

    