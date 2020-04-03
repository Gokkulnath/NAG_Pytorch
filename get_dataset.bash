
# Works well with Unix based Systems only.
# Assumes unzip is already available.

mkdir ILRVRC
cd ILRVRC
wget https://archive.org/download/Imagenet_NAG/train.zip
wget https://archive.org/download/Imagenet_NAG/valid.zip
unzip -q train.zip
unzip -q valid.zip

python Verify_Dataset.py

rm train.zip valid.zip