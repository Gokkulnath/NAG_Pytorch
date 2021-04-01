from tqdm import tqdm
import torch
import torch.nn.functional as F
import string,datetime,random
from nag.utils import get_device,get_preds,validate_generator
from nag.loss import fooling_objective,diversity_objective

base_path = ''
def save_checkpoint(model, to_save, filename='checkpoint.pth'):
    """Save checkpoint if a new best is achieved"""
    global base_path
    if to_save:
        print ("=> Saving a new best")
        torch.save(model.state_dict(), f"{base_path}/{filename}")  # save checkpoint
    else:
        print ("=> Validation Accuracy did not improve")
        
def save_perturbations(noise,arch,epoch,wabdb_flag=False):
    global base_path
    perturbations=noise.permute(0,2,3,1).cpu().detach().numpy()*255
    np.save(f'{base_path}/Perturbations_{arch}_{epoch}.npy', perturbations)
    for perturb_idx,perturbation in enumerate(perturbations[:,]):
        im = Image.fromarray(perturbation.astype(np.uint8))
        if wabdb_flag:
            wandb.log({"noise": [wandb.Image(im, caption=f"Noise_{arch}_{epoch}_{perturb_idx}")]})
        im.save(f'{base_path}/Perturbations_{arch}_{epoch}_{perturb_idx}.png')        



def fit(nb_epochs,D_model,dls,optimizer,G_adversary,device=get_device(),use_wandb=False):
    global base_path
    # Set the Discriminator in Eval mode; Weights are fixed.
    arch = 'vgg19'
    rand_strs= ''.join( random.choice(string.ascii_letters) for i in range(6))
    base_path = f"{arch}-{rand_strs}"
    os.makedirs(base_path,exist_ok=True)

    train_dl,val_dl = dls
    bs = train_dl.batch_size
    train_num =  len(train_dl.dataset)
    D_model=D_model.to(device)
    D_model.eval()
    G_adversary.train()

    timestamp=datetime.datetime.now().strftime("%d%b%Y_%H_%M")
    train_log = open(f'train_log_{D_model.__class__}_{timestamp}.txt','w')
    best_fool = 0

    for epoch in tqdm(range(nb_epochs),total=nb_epochs):
        running_loss=0
        
        train_log.writelines(f"############### TRAIN PHASE STARTED : {epoch}################")
        for batch_idx, data in tqdm(enumerate(train_dl),total =  train_num//bs):
            # Move Data and Labels to device(GPU)
            images = data[0].to(device)
            labels = data[1].to(device)
            optimizer.zero_grad()

            
            # Generate the Adversarial Noise from Uniform Distribution U[-1,1]
            latent_seed = 2 * torch.rand(bs, G_adversary.nz, 1, 1, device=device) -1 # (r1 - r2) * torch.rand(a, b) + r2
            noise = G_adversary(latent_seed)
            # print(noise.abs().max()*255,images.abs().max())

            # XB = images
            #preds_XB = f(images)
            prob_vec_clean = F.softmax(D_model(images),dim=0) # Variable q
            clean_preds ,clean_idx = get_preds(prob_vec_clean,return_idx=True,k=1)
            
            #XA = images+noise
            #preds_XA = f(images + noise)
            prob_vec_no_shuffle = D_model(images + noise)  
            qc_ =  F.softmax(prob_vec_no_shuffle,dim=0).gather(1,clean_idx) # Variable q'c

            # 1. fooling_objective: encourages G to generate perturbations that decrease confidence of benign predictions
            fool_obj, mean_qc_ = fooling_objective(qc_)
            # Perturbations  are shuffled across the batch dimesion to improve diversity
            #XS = images+ noise[torch.randperm(bs)]
            prob_vec_shuffled =   D_model(images + noise[torch.randperm(bs)])
            
            # 2.  encourages Generator to explore the space of perturbations and generate a diverse set of perturbations
            divesity_obj=diversity_objective(prob_vec_no_shuffle, prob_vec_shuffled)

            # Compute Total Loss
            total_loss = divesity_obj + fool_obj
            
            # Lets perform Backpropagation to compute Gradients and update the weights
            total_loss.backward()
            optimizer.step()
            
            # wandb Logging : Expensive : Logs Perturbation Images each iteration
            # perturbations=noise.permute(0,2,3,1).cpu().detach().numpy()*255
            # for perturb_idx,perturbation in enumerate(perturbations[:,]):
            #     im = Image.fromarray(perturbation.astype(np.uint8))
            #     wandb.log({"noise": [wandb.Image(im, caption=f"Noise_{arch}_{epoch}_{perturb_idx}")]})
            if use_wandb:
                wandb.log({
                    "fool_obj": fool_obj.item(),
                    "divesity_obj": divesity_obj.item(),
                    "total_loss":total_loss.item(),
                })
            
            running_loss += total_loss.item()
            
            if batch_idx!=0  and batch_idx % 1000 ==0 :
                train_log.writelines(f"############### VALIDATION PHASE STARTED : {epoch}, Step : {int(batch_idx / 100)} ################\n")
                fool_rate,total_fool= validate_generator(noise,D_model,val_dl)
                print(f"Fooling rate: {fool_rate}. Total Items Fooled :{total_fool}")
                train_log.writelines(f"Fooling rate: {fool_rate}. Total Items Fooled :{total_fool}\n")
                if fool_rate>best_fool:
                    best_fool=fool_rate
                    save_perturbations(noise,arch,epoch,wabdb_flag=use_wandb)
        print(f"Diversity Loss :{divesity_obj.item()} \n Fooling Loss: {fool_obj.item()} \n")
        print(f"Total Loss after Epoch No: {epoch +1} - {running_loss/(train_num//train_dl.batch_size)}")
        train_log.writelines(f"Loss after Epoch No: {epoch +1} is {running_loss/(train_num//train_dl.batch_size)}\n")
        save_checkpoint(G_adversary, to_save= True, filename=f'GeneratorW_{arch}_{epoch}.pth')
    train_log.close()