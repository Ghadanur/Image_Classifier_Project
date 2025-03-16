# Image Classifier Project  

## Overview  
This project implements an image classifier using a deep learning model trained on the [102 Category Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/). The goal is to develop a neural network capable of classifying images into different flower species.  

## Motivation  
Initially, training was performed on provided Workspace, but due to several issues, the training process was moved to **Kaggle** for better computational resources and efficiency. The implementation is inspired by Python resources, Udacity course materials, and various GitHub repositories.  

## References  
The project takes inspiration and guidance from various users on GitHub
## Features  
- Uses a **pretrained model** for feature extraction.  
- Implements **transfer learning** for better accuracy.  
- Includes **data augmentation** for improved generalization.  
- Trains and evaluates the model on **Kaggle GPUs**.  
- Exports the trained model for inference.  

## Installation  
To run the project, install the required dependencies:  
```bash
pip install torch torchvision numpy matplotlib pandas
```

## Training  
1. Load the dataset and preprocess the images.  
2. Define a convolutional neural network (CNN) using a pretrained model.  
3. Train the model using **PyTorch**.  
4. Evaluate the model on test data.  

## Results  
- Achieved more than **70% accuracy** on the validation set.  
- Model generalizes well to unseen images.  

## Future Improvements  
- Fine-tune the model for better accuracy.  
- Deploy as a **web app** for real-time predictions.  
