# [HKU] COMP3340 Group Project

## 1. Project Requirement

Image classification is one of the core problems in computer vision that, despite its simplicity, has a large variety of practical applications. In this project, our group want to train a convolutional Neural Network (CNN) or Transformer, to recognize flower types. 

## 2. Project Overview

**OUR WORK**

As a pre-requisite, we evaluated baseline models including AlexNetHere is one of the most basic CNN named AlexNet (Source Link), VGG (Source Link), ResNet (Source Link), and InceptionNet (Source Link).

The proceeding motivation from us was given flower images in 17 classes, some classes are similar in color or shape, which could be considered as one Fine-Grained Visual Classification (FGVC) Problem. It is very challenging since differences between different sub-classes of the same class are very small, either in colors, shapes, etc. High similarities make the task quite hard. 

<img width="1000" alt="Screenshot 2023-05-15 at 13 21 38" src="https://github.com/WuKunhuan/HKU_COMP3340_Group_Project/assets/79775708/e1d66cc4-75d7-44ae-9430-1395bef9913a">

We trained advanced models in FGVC in recent years to improve the performance of your network, including Attentional VGG, VAN (Visual Attention Network), ViT (Visual Transformer), Swin Transformer and TNT (Transformer in Transformer). 

By hyper-tuning the parameters, we realized in 50 epoches (our training capability capped the number of epoches), the best validation accuracy (top 1) was 0.801 for swin transformer, much lower than that for ResNet18 (0.860). Advanced models did not necessarily take more time to train, even though AlexNet was the most fastest trained models when batch size = 64. 

**LIMITATIONS**

We concluded several reasons why FGVC models do not work well in the current setting. 

- The Oxford 17 dataset was relatively small (compared to Oxford 102 flowers, Standford Dogs, ImageNet, etc.). The bigger the model is, the longer the time it takes before the model outperforms others. For example, from ResNet18 to ResNet50, we observed a decrease of validation accuracy (top 1) from 0.860 to 0.632, when hyper-parameter settings were the same. 

- Out hyper-tuning methods were limited in grid searching learning rates & batch sizes. Even though Adam optimizer & Cross Entropy loss were very common in training neural networks, it will be worth to consider the alternatives. 

- In addition, our augmented dataset was made primitively on various image processing methods, without specific purpose. It turned out that only increase the number of training data may not work well in this case. 

- Due to limited training capabilities, only 50 or 100 epoches were applied during our training, which may make some of our models' validation accuracies not converge in the end. 

**FUTURE WORK**

Dspite the limitation, we had good insight with dataset loading, model training and evaluation throughout the project. Our API developed here fit in training other models, easy to use, which make our work easily to be extensed. We will provide better models, hyper-tuning methods, or datasets when available. 

## 3. Setup Procedures

1. Download all the files. 

2. There is a couple of ways that you can setup the project. 

- As Colab Users, upload all the files into the colab. 

- As Google Drive Colab Users, upload all the files into the Google Drive. 

- As other users, just skip this step. 

3. Open the Notebooks, and follow the instructions. 

**FOR HKU COMP3340 CLASSMATES** (this section will be completed in the future)

* Follow the <a href="https://github.com/WuKunhuan/HKU_COMP3340/tree/main/HKU%20CS%20GPU%20Farm" target="_blank">HKU CS GPU Farm Setup Guide</a>, or 

* Follow the <a href="https://github.com/WuKunhuan/HKU_COMP3340/tree/main/Local%20Environment%20Setup" target="_blank">Local Environment Setup Guide</a>
