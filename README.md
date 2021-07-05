# MultitaskOCTA
This repository is an official PyTorch implementation of paper: 

"BSDA-Net: A Boundary Shape and Distance Aware Joint Learning Framework for Segmenting and Classifying OCTA Images", MICCAI 2021. 

"Multi-task Learning Based Ocular Disease Discrimination and FAZ
Segmentation Utilizing OCTA Images", EMBC 2021[Coming soom]. 

MICCAI 2021



EMBC 2021 [Coming soom]


## Dependencies

### Packages
* Python 3.7
* PyTorch >= 1.7.0
* Numpy
* Sklearn
* Segmentation Models Pytorch
* TensorboardX
* OpenCV
* numpy
* Tqdm
* surface-distance

### Data Preprocessing
Using the file *pre_dis.m* in Matlab formula for image preprocessing to generate *Boundary Heatmaps* and *Signed distanced maps (SDMs)* for training BSDA-Net. 

Run *octaaug.py* to start preprocessing automatically with preset directory value to make augment OCTA images, which is stored in local directory.

## Directory Structure
```bash
├── contour
   └── 1.png
   └── 2.png
   ...
├── dist_contour
   └── 1.mat
   └── 2.mat
   ...
├── dist_mask
   └── 1.mat
   └── 2.mat
   ...
├── dist_signed_01
   └── 1.mat
   └── 2.mat
   ...
├── dist_signed_11 (used in the MICCAI paper)
   └── 1.mat
   └── 2.mat
   ...
├──image
   └── 1.jpg
   └── 2.jpg
   ...
├── mask
   └── 1.jpg
   └── 2.jpg
   ...
```

## Training Code
To start training, you should set the parameters used for training:
* train_path: Training image path.
* val_path: Validation image path.
* test_path: Testing image path.
* save_path: Path for saving results.
* train-type: Training type, including single classification & segmentation, cotraining or multitask.
* model_type: Used for single segmentation, cotraining or multitask. The segmentation architecture used for training. 
* batch_size: Batch size for training stage.
* val_batch_size: Batch size for validation. 
* num_epochs: Total number of epochs for training stage. 
* use_pretrained: Use pretrained weight on ImageNet or not. 
* loss_type: Loss used for training stage. 
* LR_seg: Learning rate setting for segmentation process. 
* LR_clf: Learning rate setting for classification process. 
* classnum: Used for single classification, cotraining or multitask. Class number for classification. 

For simply start training, you can use our preset shell file named *Demo.sh* with prepared dataset stored in local path. Or you can set the parameters listed above to define your own training architecture. The results will be stored at the local path with your dataset name as a folder named as *model_type+loss_type*. 

## Testing code
To start testing, you should set the parameters that is used for training to load the model file correctly: 
* train_path: Training image path.
* val_path: Validation image path.
* test_path: Testing image path.
* save_path: Path for saving results.
* train-type: Training type, including single classification & segmentation, cotraining or multitask.
* model_type: Used for single segmentation, cotraining or multitask. The segmentation architecture used for training. 
* val_batch_size: Batch size for validation. 
* use_pretrained: Use pretrained weight on ImageNet or not. 
* loss_type: Loss used for training stage. 

## Results
From left to right, they are respectively representation of segmentation results of FAZ using different network architecture. The bottom line represents corresponding boudnary heatmaps and signed distance maps for groundtruth. 



## Citation
L. Lin$\dag$, Z. Wang$\dag$, J. Wu, Y. Huang, J. Lyu, P. Cheng, J. Wu, X. Tang*, "BSDA-Net: a boundary shape and distance aware joint learning framework for segmenting and classifying OCTA images",  In the 24th International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI), Strasbourg, France, September 2021.
## Contact
Li Lin (linli55@mail2.sysu.edu.cn)

Zhonghua Wang (Wzhjerry1112@gmail.com)

## Acknowledgements
Thanks for [segmentation models pytorch](https://github.com/qubvel/segmentation_models.pytorch) for the implementation of cotraining codes. 
