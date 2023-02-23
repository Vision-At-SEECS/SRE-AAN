# Squeeze-and-Residual-Excitation based Aggregated Attention Network for Improved Super-Resolution in Remote Sensing Imagery
Squeeze-and-Residual-Excitation based Aggregated Attention Network (SRE-AAN) improves super-resolution (SR) on remote-sensing imagery compared to other state-of-the-art attention-based SR models.

The model is built in PyTorch.

## Contents
1. [Introduction](#introduction)
2. [Network](#network)
3. [Setup](#setup)
4. [Test](#test)
5. [Results](#results)
6. [Authors](#authors)

## Introduction

Super-resolution (SR) is a promising technique for enhancing the quality of remote sensing imagery, which can improve the accuracy of various computer vision tasks such as object detection, classification, and segmentation. Extensive research has been conducted on SR for natural and remote sensing imagery, and deep convolutional neural networks (CNNs) have shown remarkable progress in this field. Attention mechanisms are widely used in deep CNNs because they enable the deep-learning models to assign weights to the important areas of the feature map. In this paper, we propose the Squeeze-and-Residual-Excitation (SRE) attention block that incorporates attention across the channels of the feature map but also scales the overall map and uses the resulting feature map as a residual, resulting in an improvement in performance.
We also propose an SR framework by utilizing multiple attention blocks keeping the SRE attention module at the core of these blocks for enhancing the spatial resolution of remote sensing imagery.
Our proposed network, the Squeeze-and-Residual-Excitation based Aggregated Attention Network (SRE-AAN), outperforms other attention-based deep SR models for 4x- and 8x-upsampling on two remote sensing datasets: Satellite Imagery Multi-Vehicles Dataset (SIMD), which comprises 5000 high-resolution remote sensing images, and DOTA, a large-scale satellite imagery dataset. Furthermore, we demonstrate the applicability of our proposed SR framework to object detection and report improvements in the performance of the YoloV5 object detection model when used on low-resolution images. We perform several experiments to evaluate the efficacy of our proposed SR framework for object detection on the SIMD dataset.

## Network

![SRE-HAN Super Resolution Framework](/figures/sre_aan_complete.png)

## Setup
The setup can be done through one  of the following methods:

Create a new conda environment and install the following packages:

1. Python (conda env)
2. Pytorch with Cuda (use command for conda from Pytorch website)
3. scipy
4. scikit-image
5. h5py
6. opencv
7. matplotlib
8. tqdm

Use the new environment for testing and performing experiments.

## Test

Place images that you want to upsample in the 'demo/low_res_images' folder

CD to 'src' and run one of the following script

The upsampled images will be found in the 'demo/results/results-Demo' folder.

```bash
#for 4x upsampling
python main.py --template 4X_SRE_AAN --pre_train ../trained_models/sre_aan_x4.pt --n_GPUs=2 --data_test Demo --dir_demo ../demo/low_res_images --test_only --save ../demo/results --save_results

#for 8x upsampling
python main.py --template 8X_SRE_AAN --pre_train ../trained_models/sre_aan_x8.pt --n_GPUs=2 --data_test Demo --dir_demo ../demo/low_res_images --test_only --save ../demo/results --save_results
```

## Results

### SR Results

The super-resolution performance over the Satellite Imagery Multi-Vehicles Dataset of our model and three previous state-of-the-art models are shown in this section. Our model outperforms the other models for both 4x and 8x upsampling.

![Results](/figures/results.png)

### Object Detection 8x SR Results

We use YoloV5-Medium mopel presented by Glenn Jocher to perform two types of object detection experiments.

#### Experiment 1:
YoloV5-Medium is trained and evaluated on the upsampled versions of the dataset obtained through using the super-resolution models. The results are as follows:

![Detection Results](/figures/Detection_Results.png)

#### Experiment 2:
YoloV5-Medium is trained over the ground truth dataset and then evaluated on the upsampled versions of the test set obtained through using the SR models.

![Detection Results2](/figures/detection_results2.png)

### Visual Results for 8x Upsampling and Consequent Object Detection

The visual results for the performance of the object detection model that has been trained on the ground truth dataset and then evaluated on the images upsampled through bicubic method and SRE-HAN model are shown here. 

![Visual Detection Results](/figures/8x_visual_detection_results.png)
Ground truth trained YoloV5 Object Detection Model is applied over both of the above images and we can see that the detection model is unable to detection any of the vehicles in the bicubic upsampled image. On the other hand, Yolov5 detects many of the objects in the image upsampled through our SRE-HAN model.

![Visual Detection Results2](/figures/8x_visual_detection_results2.png)
Similarly, in the bicubic upsampled image the detection model detects only 1 object, whereas in the SRE-HAN upsampled image all of the objects have been detected except one.

## Authors

Bostan Khan <bkhan.mscs19seecs@seecs.edu.pk>

Adeel Mumtaz <adeelmumtaz@gmail.com>

Zuhair Zafar <zuhair.zafar@seecs.edu.pk>

Muhammad Moazam Fraz <mfraz@turing.ac.uk>