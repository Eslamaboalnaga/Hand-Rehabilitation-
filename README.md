# Hand Rehabilitation Detection Using Computer Vision

## Overview
This is my graduation project. We utilize AI-based methods of pose estimation to address peripheral in a mobile application.
We created the first dataset for this problem and implement it using Machine learning and Deep learning.

Models : [Convolutional Neural Network – Linear Regression – Random Forest Classifier ].

# ColorHandPose3D network

## Usage: Forward pass
The network ships with a minimal example, that performs a forward pass and shows the predictions.

- Download [data](https://lmb.informatik.uni-freiburg.de/projects/hand3d/ColorHandPose3D_data_v3.zip) and unzip it into the projects root folder (This will create 3 folders: "data", "results" and "weights")
- *run.py* - Will run a forward pass of the network on the provided examples

You can compare your results to the content of the folder "results", which shows the predictions we get on our system.


## Recommended system
Recommended system (tested):
- Ubuntu 16.04.2 (xenial)
- Tensorflow 1.3.0 GPU build with CUDA 8.0.44 and CUDNN 5.1
- Python 3.5.2


Python packages used by the example provided and their recommended version:
- tensorflow==1.3.0
- numpy==1.13.0
- scipy==0.18.1
- matplotlib==1.5.3

## Preprocessing for training and evaluation
In order to use the training and evaluation scripts you need download and preprocess the datasets.

### Rendered Hand Pose Dataset (RHD)

- Download the dataset accompanying this publication [RHD dataset v. 1.1](https://lmb.informatik.uni-freiburg.de/resources/datasets/RenderedHandposeDataset.en.html)
- Set the variable 'path_to_db' to where the dataset is located on your machine
- Optionally modify 'set' variable to training or evaluation
- Run

# The Datasets
We created the first dataset for this problem by deep learning model.
We use a model consists of  Convolutional Neural Network estimating 2D Hand Pose from a single RGB Image.
The main purpose of using this model is to collecting a set of hand key points.
![image](https://user-images.githubusercontent.com/57270149/151982420-531316a0-a5a9-467b-9d8a-d042dcc05af9.png)

- The hand is localized within the image by a segmentation network (HandSegNet). 
- Accordingly to the hand mask, the input image is cropped and serves as input to the PoseNet. 
- This localizes a set of hand keypoints represented as score maps c. Subsequently, the PosePrior network estimates the most likely 2D structure conditioned on the score maps.

# Machine Learning Models
- Linear Regression is the best evaluation algorithm for this dataset.

![image](https://user-images.githubusercontent.com/57270149/151983689-22fb0047-d69e-40a0-a0ff-7ebe4c1e0ba6.png)

- Random Forest Classifier is the best evaluation algorithm for knowing the hand left or right.


