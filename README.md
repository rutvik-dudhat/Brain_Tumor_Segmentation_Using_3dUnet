
# Brain Tumor Segmentation using 3D U-Net  

## Overview  
This project implements a **3D U-Net model** for brain tumor segmentation using the **BraTS2020 dataset**. The model leverages advanced deep learning techniques to identify and segment tumor regions in MRI scans, with a focus on enhancing segmentation accuracy and robustness.  

## Features  
- **3D U-Net architecture** for volumetric segmentation.  
- **Pre-trained ResNet34 backbone** for robust feature extraction.  
- Hybrid loss function: **Dice + Focal loss** to address class imbalance.  
- Comprehensive evaluation using **Dice coefficient, precision, and recall**.  
- **Data augmentation** techniques for improved generalization.  
- Cross-validation for robust performance assessment.  

## Dataset  
The project uses the **BraTS2020 dataset**, which includes:  
- Multi-parametric MRI scans: T1, T1-CE, T2, and FLAIR.  
- Ground truth labels for three tumor regions: enhancing tumor, tumor core, and whole tumor.  

> **Note:** Ensure you comply with the dataset’s terms and conditions when downloading and using it.  

## Model Performance  
| Model     | Mean IoU | Dice Score | Sensitivity | Specificity |
|-----------|----------|------------|-------------|-------------|
| ResNet34  | 0.2386   | 0.3002     | 0.4201      | 0.8505      |
| 3D U-Net  | 0.2750   | 0.3305     | 0.4500      | 0.8700      |


## Prerequisites  
- Python 3.10 or later  
- TensorFlow >= 2.4.0  
- NumPy  
- Matplotlib  
- Scikit-learn  


## Installation  
1. Clone the repository:  
   ```bash
   git clone https://github.com/rutvik-dudhat/Brain_Tumor_Segmentation_Using_3dUnet.git  
