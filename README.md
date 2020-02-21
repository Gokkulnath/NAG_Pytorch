# NAG - Network for Adversary Generation [CVPR 2018](http://val.serc.iisc.ernet.in/nag/) WIP
Pytorch implementation of NAG : Network for adversary generation 

Konda Reddy Mopuri*, Utkarsh Ojha*, Utsav Garg, R. Venkatesh Babu.

This work is an attempt to explore the manifold of perturbations that can cause CNN based classifiers to behave absurdly. At present, this repository provides the facility to train the generator that can produce perturbations to fool VGG F, VGG 16, VGG 19, GoogleNet, CaffeNet, ResNet 50, ResNet 152. The generator architecture has been modified from here.

Architecture
![](resources/nag.png)


## Training your own generator


## Testing on clean images


## Reference
```
@inproceedings{nag-cvpr-2018,
  title={NAG: Network for Adversary Generation},
  author={Mopuri, Konda Reddy and Ojha, Utkarsh and Garg, Utsav and Babu, R Venkatesh},
 booktitle = {Proceedings of the IEEE Computer Vision and Pattern Recognition ({CVPR})},
 year = {2018}
}
```
