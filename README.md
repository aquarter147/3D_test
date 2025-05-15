# 3D Vision Test: Implement NERF from scratch (100 points)
Goals:  In this assigment, you will implement the NERF model from scratch base on the paper [NERF](https://arxiv.org/abs/2003.08934)

## 0. Setup
Packages that you will need can be installed using the 'requirements.txt' file (`pip install -r requirements.txt`).

## 1. Dataset
The Fox dataset is a small synthetic dataset of a 3D fox model rendered from multiple viewpoints. It's commonly used in NeRF tutorials due to its simplicity, small size, and fast training time — ideal for quick experiments and learning. Download the fox dataset from [here](https://drive.google.com/file/d/1hpQn5Fe92a1MCXstZZYVneqKLkIQTL2x/view?usp=sharing)

## 2. Assingments

### 2.1 Render (30 pts)
Read the paper about the rendering technique and complete the TODO code

### 2.2 Model (30 pts)
Read the paper about the model architecture and complete the TODO code

### 2.3 Train the model
You will need to train the model by running:

`python nerf_train.py`

### 2.4 Optimize (40 pts)
Run the test.py to test model. Optimize the model to have the best results with PSNR metric. 

## 3. Report
After doing the implementation of NERF, report your results in a .pdf file. 

