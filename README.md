# Simpsons Character Classifier (CNN)

This project implements a Convolutional Neural Network (CNN) trained to classify characters from *The Simpsons*.

## Overview
- Trained on **16,000 images** of Simpsons characters  
- Classifies **20 distinct characters**
- Achieves approximately **90% accuracy** on test dataset

## Dataset
The dataset is sourced from Kaggle: https://www.kaggle.com/datasets/alexattia/the-simpsons-characters-dataset

The original author included test images different from training set. Trained model achieved around 90% accuracy.

## Project Structure
- **train.ipynb** — Walks through the full training pipeline step by step  
- **train.py** — Trains the model directly without using a notebook
- **model.py** — Defines 4 layer convolution neural network
- **inference.ipynb** — Runs inference and generates prediction results in a `.json` file for evaluation
- **simpsons_cnn4conv.pth** — Saved trained model for reuse
