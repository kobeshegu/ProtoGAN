# Readme.md

## Source codes for ProtoGAN: Towards High Diversity and Fidelity Image Synthesis under limited Data

### Requirements

* imageio==2.9.0
* lmdb==1.2.1
* opencv-python==4.5.3
* pillow==8.3.2
* scikit-image==0.17.2
* scipy==1.5.4
* tensorboard==2.7.0
* tensorboardx==2.4
* torch==1.7.0+cu110
* torchvision==0.8.1+cu110
* tqdm==4.62.3

### Run the code

Use ./run.sh to run the code.

The results and models will be automatically saved in /train_resutls folder.

The results of FID and IS will be shown in FID.txt and IS.txt, respectively.

You can train our ProtoGAN with your own datasets by setting the adequate path and change the corresponding resolution for your data, enjoy!

### Important notes 
1. Our code is heavily developed based on [FastGAN](https://github.com/odegeasslbc/FastGAN-pytorch). We thanks a lot for their great work. The code is for research only. 
2. Feel free to contact me at kobeshegu@gmail.com if you have any questions.