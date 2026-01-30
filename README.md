# Dental Image Segmentation using Deep Learning

## 📌 Introduction
Dental image segmentation is a critical task in computer-aided dental diagnosis, treatment planning, and forensic dentistry. Manual segmentation of teeth from oral images is time-consuming and prone to human error. This project focuses on developing an automated deep learning–based system for accurate dental image segmentation using state-of-the-art semantic segmentation models.

---

## 🎯 Objectives
1. To study deep learning–based image segmentation techniques for dental image analysis.  
2. To implement and compare multiple segmentation models on dental images.  
3. To develop an automated computational model for accurate tooth region segmentation.  

---

## 🗂 Dataset Description
- Input images: RGB dental images (cropped mouth region)
- Ground truth: Binary segmentation masks created using **LabelMe**
- Format:
  - Images: `.jpg`
  - Masks: `.png`
- Dataset split:
  - Training set
  - Validation set
  - Test set (unseen during training)

---

## 🔄 Methodology / Workflow
1. Data collection and annotation (JSON → mask conversion)
2. Image preprocessing and augmentation
3. Dataset splitting (train / validation / test)
4. Model training
5. Model evaluation using quantitative metrics
6. Batch prediction and result visualization
7. Model comparison and analysis

---

## 🧠 Models Implemented
- **DeepLabV3+ (ResNet50 / ResNeXt101 backbone / ResNet101 varied output stride)**
- **Attention U-Net**
- **Ensemble Model (DeepLabV3 + Attention U-Net)**

---

## 📊 Evaluation Metrics
The models were evaluated using:
- Dice Coefficient
- Intersection over Union (IoU)
- Pixel Accuracy
- Precision
- Recall

---

## 🏆 Results Summary

| Model | Dice Score | IoU | Pixel Acc | Precision | Recall
|------|-----------|-----|
| DeepLabV3+ (ResNeXt50) | ~0.97 | ~0.94 | ~0.99 | ~0.98 | ~0.96
| DeepLabV3+ (ResNeXt101, o/p stride=16) | ~0.98 | ~0.96 | ~0.99 | ~0.99 | ~0.98
| DeepLabV3+ (ResNeXt101, o/p stride=8, weighted loss) | ~0.99 | ~0.97 | ~0.99 | ~0.98 | ~0.99
| Attention U-Net | ~0.96 | ~0.92 | ~0.98 | ~0.96 | ~0.96
| Ensemble Model | ~0.97 | ~0.95 | ~0.99 | ~0.98 | ~0.97

> DeepLabV3+ with ResNeXt101 backbone achieved the best overall performance.

---

## 🖼 Qualitative Results
Segmented masks for each model are saved in separate folders:

batch_results/
├── deeplab/
├── attention_unet/
├── ensemble/
├── resnet101/



