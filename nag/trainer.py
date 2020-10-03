from tqdm import tqdm



def fit(nb_epochs,D_model,dls,optimizer,adversarygen=adversarygen):
    # Set the Discriminator in Eval mode; Weights are fixed.
    train_dl,val_dl = dls
    D_model=D_model.to(device)
    D_model.eval()
    timestamp=datetime.datetime.now().strftime("%d%b%Y_%H_%M")
    train_log = open(f'train_log_{arch}_{timestamp}.txt','w')
    for epoch in tqdm(range(nb_epochs),total=nb_epochs):
        running_loss=0
        rand_str= ''.join( random.choice(string.ascii_letters) for i in range(6))
        
        train_log.writelines(f"############### TRAIN PHASE STARTED : {epoch}################")
        for batch_idx, data in tqdm(enumerate(train_dl),total = train_num//train_dl.batch_size):
            # Move Data and Labels to device(GPU)
            images = data[0].to(device)
            labels = data[1].to(device)

            
            # Generate the Adversarial Noise from Uniform Distribution U[-1,1]
            latent_seed = 2 * torch.rand(bs, nz, 1, 1, device=device,requires_grad=True) -1 # (r1 - r2) * torch.rand(a, b) + r2
            noise = adversarygen(latent_seed)
            optimizer.zero_grad()

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
#             perturbations=noise.permute(0,2,3,1).cpu().detach().numpy()*255
#             for perturb_idx,perturbation in enumerate(perturbations[:,]):
#                 im = Image.fromarray(perturbation.astype(np.uint8))
#                 wandb.log({"noise": [wandb.Image(im, caption=f"Noise_{arch}_{epoch}_{perturb_idx}")]})
            wandb.log({"fool_obj": fool_obj.item(),
                       "divesity_obj": divesity_obj.item(),
                       "total_loss":total_loss.item(),
                      })        
            
            running_loss += total_loss.item()
            
            if batch_idx!=0  and batch_idx % 100 ==0 :
                train_log.writelines(f"############### VALIDATION PHASE STARTED : {epoch}, Step : {int(batch_idx / 100)} ################")
                fool_rate,total_fool= validate_generator(noise,D_model,val_dl)
                print(f"Fooling rate: {fool_rate}. Total Items Fooled :{total_fool}")
                train_log.writelines(f"Fooling rate: {fool_rate}. Total Items Fooled :{total_fool}")
        print(f"Diversity Loss :{divesity_obj.item()} \n Fooling Loss: {fool_obj.item()} \n")
        print(f"Total Loss after Epoch No: {epoch +1} - {running_loss/(train_num//train_dl.batch_size)}")
        train_log.writelines(f"Loss after Epoch No: {epoch +1} is {running_loss/(train_num//train_dl.batch_size)}")
        # to_save can be any expression/condition that returns a bool
        
        save_checkpoint(adversarygen, to_save= True, filename=f'GeneratorW_{arch}_{epoch}_{rand_str}.pth') 
        if epoch % 1 == 0:
#             save_perturbations(noise,arch,epoch)
            save_perturbations(noise,arch,epoch,wabdb_flag=True)
    train_log.close()