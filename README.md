# Plant Disease Detection Lightweight Framework

![Project Banner](https://your-image-url.com/banner.png)

[![Build Status](https://img.shields.io/github/actions/workflow/status/Dhanushrajashekar/Plant_Disease_Detection_LightWeight_Framework/CI.yml?branch=main)](https://github.com/Dhanushrajashekar/Plant_Disease_Detection_LightWeight_Framework/actions)
[![License](https://img.shields.io/github/license/Dhanushrajashekar/Plant_Disease_Detection_LightWeight_Framework)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/Dhanushrajashekar/Plant_Disease_Detection_LightWeight_Framework)](https://github.com/Dhanushrajashekar/Plant_Disease_Detection_LightWeight_Framework/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/Dhanushrajashekar/Plant_Disease_Detection_LightWeight_Framework)](https://github.com/Dhanushrajashekar/Plant_Disease_Detection_LightWeight_Framework/network)

A lightweight framework for detecting plant diseases using machine learning. This tool helps farmers, agronomists, and researchers in early detection and management of plant diseases, improving crop yields and reducing losses.

## Table of Contents
- [Introduction](#introduction)
- [Model Deployment](#model-deployment)
- [Metadata](#metadata)
- [Models](#models)
- [Dataset](#dataset)
- [Contributions](#contributions)
- [Repository Structure](#repository-structure)
- [Usage](#usage)
- [Dependencies and Setup](#dependencies-and-setup)
- [License](#license)

## Introduction

Plant diseases can cause significant damage to crops, leading to substantial economic losses and food security issues. Early detection and accurate diagnosis are crucial for effective disease management. This project leverages advanced machine learning techniques to develop a model capable of identifying different plant diseases from leaf images.

The framework is designed to be lightweight, making it suitable for deployment on devices with limited computational resources, such as mobile phones and edge devices. The model has been trained on a custom dataset that includes various classes of plant diseases and healthy leaves.

## Model Deployment

The model is deployed on Hugging Face Spaces. You can interact with the model and see it in action using the link below:

[![Hugging Face Spaces](https://img.shields.io/badge/Hugging%20Face-Spaces-orange)](https://huggingface.co/spaces/Cringe1324/Plant_Leaf_Disease_Detection_LightWeight_Model)

## Metadata

| title | emoji | colorFrom | colorTo | sdk | sdk_version | app_file | pinned |
|-------|-------|-----------|---------|-----|-------------|----------|--------|
| Plant Leaf Disease Detection LightWeight Model | 🐱 | green | gray | gradio | 4.38.1 | app.py | false |

## Models

The models used in this framework are custom-trained models specifically designed for plant disease detection.

## Dataset

### Current Updated Dataset Used:

In this dataset, 38 different classes of plant leaf and background images are available. The dataset is split into a 90-10 ratio for training and testing.

- **Total Images**: 47,179
  - **Training File**: 39,379 images
  - **Validation File**: 3,900 images
  - **Test File**: 3,900 images

This dataset is a subset of the original dataset and has been used to train the custom models deployed in this framework.

## Contributions

In this project, we compared the performance of several models to determine the best one for plant disease detection. The models compared include:

- **ResNet18**
- **MobileViT**
- **GhostNetV2**

### Results

The models were evaluated based on their accuracy and computational efficiency. The best-performing model was then selected for deployment.

### Deployment

The selected model was deployed by developing a graphical user interface (GUI) and hosting it on Hugging Face Spaces. This allows users to easily interact with the model and obtain predictions for plant disease detection.

## Repository Structure

- `hugging_Face_Deployment/`: Contains deployment scripts and configurations for the Hugging Face Space.
- `Models/`: Pre-trained models and training scripts.
- `Test_final/`, `Train_final/`, `Val_final/`: Datasets used for training, validation, and testing.

## Usage

To use this repository and run the code locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Dhanushrajashekar/Plant_Disease_Detection_LightWeight_Framework.git
   cd Plant_Disease_Detection_LightWeight_Framework

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
