
# Works well with Unix based Systems only.
# Assumes unzip is already available.

mkdir ILRVRC
cd ILRVRC
wget https://archive.org/download/Imagenet_NAG/train.zip
wget https://archive.org/download/Imagenet_NAG/valid.zip
unzip -q train.zip
unzip -q valid.zip

python Verify_Dataset.py

# To Get the VGG-F model weights either dowload from URL : 
pip install gdown --user
gdown https://drive.google.com/file/d/1TjaaUAex89NUOBihFxvrBFuHRfv8Jd5Y/view?usp=sharing
rm train.zip valid.zip