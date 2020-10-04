import torch
import torch.nn.functional as F
import os
from PIL import Image
import numpy as np
import warnings
try:
    import wandb
except:
    warnings.warn('Check wandb is installed. Logging Functions may not Work. If not install using the command pip install wandb. ')
from nag.model import model_dict
from tqdm import tqdm 
# epsillon=10
# batch_size=32
# latent_dim = 10
img_h,img_w,img_c=(224,224,3)
latent_dim=10
arch='resnet50'
archs=model_dict.keys() # ['vgg-f','vgg16','vgg19','googlenet','resnet50','resnet152'] 

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_bs(arch):
    if torch.cuda.is_available():
#         GPU_BENCHMARK= 8192.0
#         GPU_MAX_MEM = torch.cuda.get_device_properties(device).total_memory / (1024*1024)
#         BS_DIV= GPU_BENCHMARK/GPU_MAX_MEM
#         print(f"Current GPU MAX Size : {GPU_MAX_MEM}. {BS_DIV}")

        if arch  not in ['resnet50','resnet152']:#  ['vgg16','vgg19','vgg-f','googlenet']:
            bs=int(32)
        elif arch in ['resnet50','resnet152']:
            bs=int(16)
        else:
            raise ValueError(f'Architecture type not supported. Please choose one from the following {archs}')
    else:
        bs=8 # OOM Error
    return bs

#get_bs(arch)


def save_checkpoint(model, to_save, filename='checkpoint.pth'):
    """Save checkpoint if a new best is achieved"""
    if to_save:
        print ("=> Saving a new best")
        torch.save(model.state_dict(), filename)  # save checkpoint
    else:
        print ("=> Validation Accuracy did not improve")
        
def save_perturbations(noise,arch,epoch,wabdb_flag=False):
    rand_str= ''.join( random.choice(string.ascii_letters) for i in range(6))
    os.makedirs(f"{arch}-{rand_str}",exist_ok=True)
    perturbations=noise.permute(0,2,3,1).cpu().detach().numpy()*255
    np.save(f'{arch}-{rand_str}/Perturbations_{arch}_{epoch}.npy', perturbations)
    for perturb_idx,perturbation in enumerate(perturbations[:,]):
        
        im = Image.fromarray(perturbation.astype(np.uint8))
        if wabdb_flag:
            wandb.log({"noise": [wandb.Image(im, caption=f"Noise_{arch}_{epoch}_{perturb_idx}")]})
        im.save(f'{arch}-{rand_str}/Perturbations_{arch}_{epoch}_{perturb_idx}.png')        

# TODO 
def visualize_perturbations():
    # MAtplotlib Subplot ?
    # Subplots(4*4) or (3*3)
    # From Memory or Disk - Epoch number ?
    pass



def get_preds(predictions,return_idx=False, k=1):
    idxs= torch.argsort(predictions,descending=True)[:,:k]
    if return_idx:
        return predictions[:,idxs], idxs
    return  predictions[:,idxs]



# val_iterations = val_num/bs

def compute_fooling_rate(prob_adv,prob_real):
    '''Helper function to calculate mismatches in the top index vector
     for clean and adversarial batch
     Parameters:
     prob_adv : Index vector for adversarial batch
     prob_real : Index vector for clean batch
     Returns:
     Number of mismatch and its percentage
    '''
    nfool=0
    size = prob_real.shape[0]
    for i in range(size):
        if prob_real[i]!=prob_adv[i]:
            nfool = nfool+1
    return nfool, 100*float(nfool)/size      


def validate_generator_old(noise,val_dl,val_iterations=10):
    
    total_fool=0
    print("############### VALIDATION PHASE STARTED ################")
    train_log.writelines("############### VALIDATION PHASE STARTED ################")
    
    for val_idx in range(val_iterations):
        for batch_idx, data in enumerate(val_dl):
            images = data[0].to(device)
#             labels = data[1].to(device)
            
            prob_vec_clean = F.softmax(D_model(images),dim=0) # Variable q
            prob_vec_no_shuffle = D_model(images + noise)  
            nfool, _ = compute_fooling_rate(prob_vec_no_shuffle,prob_vec_clean)
            total_fool += nfool
    
    
    fool_rate = 100*float(total_fool)/(val_iterations*batch_size)       
    print(f"Fooling rate: {foolr}. Total Items Fooled :{total_fool}")
    train_log.writelines(f"Fooling rate: {foolr}. Total Items Fooled :{total_fool}")

    
    
def validate_generator(noise,D_model,val_dl,device=get_device()):
    total_fool=0
    for batch_idx, data in tqdm(enumerate(val_dl),total = len(val_dl.dataset)//val_dl.batch_size):
        val_images = data[0].to(device)
        val_labels = data[1].to(device)

        prob_vec_clean,clean_idx = get_preds(F.softmax(D_model(val_images),dim=0),return_idx=True) # Variable q
        prob_vec_no_shuffle,adv_idx = get_preds(F.softmax(D_model(val_images + noise),dim=0),return_idx=True)  
        nfool, _ = compute_fooling_rate(adv_idx,clean_idx)
        total_fool += nfool

    fool_rate = 100*float(total_fool)/(len(val_dl.dataset))
    return fool_rate,total_fool



