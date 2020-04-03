# **Pytorch implementation** NAG - Network for Adversary Generation 

Official Project Page: [Link](http://val.serc.iisc.ernet.in/nag/)
Authors: Konda Reddy Mopuri*, Utkarsh Ojha*, Utsav Garg, R. Venkatesh Babu.

This work is an attempt to explore the manifold of perturbations that can cause CNN based classifiers to behave absurdly. At present, this repository provides the facility to train the generator that can produce perturbations to fool VGG F, VGG 16, VGG 19, GoogleNet, CaffeNet, ResNet 50, ResNet 152. The generator architecture has been modified from here.

Architecture
![](resources/nag.png)

## Setting up Data Manually
**P.S**: For the train split we randomly sampled 10 instances from each target class as described in the paper.
Note: Dataset is Aproximately 7.5 GB in Size. Use Verify_Dataset.py to check for any errors.

For Unix Users: 

- Run setup_dataset.bash to get the dataset. Automatically runs Verify_Dataset.py

For Windows Users:

- Option 1: Download from Archive.org
  - [Archive Link](https://archive.org/embed/Imagenet_NAG)
- Option 2 : Mega Download Link for Train abd Validation data of Imagenet 2012 (Obtained from Kaggle)
  - Validation Data: [Mega Link](https://mega.nz/#!yDoTDIyD!RjN6OBA92-KLpNqDeLS3OzwmAYesEbTsiQat9hT6p6s)
  - Trainning Data: [Mega Link](https://mega.nz/#!vKY0WSDa!4aibnBkiXUrO9MkhQlLGXac7wLF5HY7O4LzfdFEaeQU) 


- The Notebook Code.ipynb contains the code required for trainning the generator
- Notebooks in the directory contains utils and code to generate and interpolate perturbations from generator weights

- Note: Pretrained weights for Googlenet, Resnet50, VGG16 and VGG19 (Trained for 30 Epochs) can be found   as a Kaggle Dataset
Link : https://www.kaggle.com/gokkulnath/nag-pytorch-pretrained
##### TODO : 

- Testing on clean images


## Reference
```
@inproceedings{nag-cvpr-2018,
  title={NAG: Network for Adversary Generation},
  author={Mopuri, Konda Reddy and Ojha, Utkarsh and Garg, Utsav and Babu, R Venkatesh},
 booktitle = {Proceedings of the IEEE Computer Vision and Pattern Recognition ({CVPR})},
 year = {2018}
}
```
