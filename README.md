# Lung Nodules Detection
## Description
Lung nodules are small lesions that may indicate early-stage lung cancer. 
Due to their subtle appearance and small size in CT scans, manual inspection by radiologists can be time-consuming and prone to missed detections.

The goal of this project is to develop an automated object detection model to accurately locate lung nodules in CT images using deep learning techniques.

In this project, I apply YOLOv11n to detect two types of lung nodules. 
The model is trained on a labeled CT scan dataset and evaluated using standard object detection metrics such as Precision, Recall, and mAP.

This work focuses on improving small object detection performance through:
- High-resolution input images
- Strong data augmentation strategies
- Optimized loss weighting for better localization accuracy

This project demonstrates the application of deep learning in medical image analysis, particularly in improving early detection support systems for lung cancer screening.

The final objective is to build a reliable detection system that can assist medical professionals in identifying potential lung nodules more efficiently.
## Dataset
- Dataset: [Lung Nodules Dataset](https://www.kaggle.com/datasets/younesselbrag/lung-nodules-detection-dataset-annotations/data)  
- Image size: 416 × 416  
- Classes: `Nodule_type1`, `Nodule_type2`  
- Training set: 239 images  
- Validation set: 41 images  

The dataset is stored in the `ct_images` directory with the following structure:
```
ct_images/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
```

## Model

After comparing different models, **YOLOv11n** performs the best.

### Performance

| Model     | Precision | Recall | mAP50 | mAP75 | mAP50-95 |
|-----------|-----------|--------|-------|-------|----------|
| YOLOv11n  | 88.3%     | 82.5%  | 91.5% | 59.1% | 53.8%    |

### Metrics Explanation

- **Precision (88.3%)**  
  Among all predicted nodules, 88.3% are true positives (i.e., correctly detected nodules), while the remaining are false positives.  
  This indicates the model has a relatively low false positive rate.

- **Recall (82.5%)**  
  The model successfully detects 82.5% of all actual nodules.  
  The remaining 17.5% are missed detections (false negatives).

- **mAP50 (91.5%)**  
  Mean Average Precision at IoU = 0.5.  
  Indicates strong detection performance when moderate localization accuracy is acceptable.

- **mAP75 (59.1%)**  
  Mean Average Precision at IoU = 0.75.  
  Reflects stricter localization performance, showing that precise bounding box alignment is more challenging.

- **mAP50-95 (53.8%)**  
  Averaged mAP across IoU thresholds from 0.5 to 0.95.  
  Provides a comprehensive evaluation of both detection and localization performance.

## Training Configuration

The model is trained with the following hyperparameters:

### Training Settings
- **epochs = 150**  
  Number of training epochs. A relatively high value ensures sufficient convergence.

- **imgsz = 1280**  
  High-resolution input to capture small nodules more effectively, which is critical in medical imaging.

---

### Optimization
- **optimizer = AdamW**  
  Adaptive optimizer with decoupled weight decay, providing stable convergence.

- **lr0 = 0.003**  
  Initial learning rate.

- **lrf = 0.01**  
  Final learning rate factor for cosine decay scheduling.

- **weight_decay = 0.0005**  
  Regularization to prevent overfitting.

---

### Loss Weights
- **box = 7.0**  
  Emphasizes bounding box regression accuracy.

- **cls = 0.5**  
  Lower weight since classification is relatively simple (only 2 classes).

- **dfl = 1.8**  
  Distribution Focal Loss weight for better localization precision.

---

### Data Augmentation
- **mosaic = 1.0**  
  Strong augmentation to improve generalization and small object detection.

- **mixup = 0.2**  
  Blends images to improve robustness.

- **copy_paste = 0.3**  
  Helps increase object diversity, useful for small datasets.

- **scale = 0.5**  
  Random scaling for better scale invariance.

---

### Performance Optimization
- **cache = "ram"**  
  Loads dataset into memory for faster training.

- **batch = -1**  
  Automatically selects the largest possible batch size.

- **workers = 2**  
  Number of data loading workers (limited by hardware).

---

