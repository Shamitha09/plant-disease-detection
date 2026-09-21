# Plant Disease Detection

A deep learning model that classifies plant leaf images into 38 disease/healthy categories using transfer learning.

## Overview
This project fine-tunes a **ResNet18** (pretrained on ImageNet) to detect crop diseases from leaf images. Built as part of ongoing research toward an IEEE conference paper in Applied ML for agriculture, alongside a companion crop yield prediction model.

## Dataset
- **Source:** [PlantVillage Dataset (Kaggle)](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- **Size:** 54,305 images
- **Classes:** 38 (14 crop species × disease states + healthy), including apple, corn, grape, potato, tomato, and more

## Model
ResNet18 with ImageNet-pretrained weights, fine-tuned via transfer learning (final block + classifier layer trained, earlier layers frozen). Built in PyTorch.

- Input size: 224×224
- Optimizer: Adam, lr=0.001, StepLR scheduler
- Data augmentation: random horizontal flip, random rotation (±15°)
- Trained for 3 epochs on a T4 GPU (Google Colab)

## Results
| Metric | Value |
|---|---|
| Validation Accuracy | **99.07%** |
| Macro F1-score | **0.986** |

Per-class precision/recall stayed at or near 1.00 for most classes, with slightly lower (but still strong, 0.87–0.95) scores on visually similar disease pairs like Corn Cercospora leaf spot / Gray leaf spot.

## How to run
```bash
pip install torch torchvision scikit-learn
python plant_disease_detection.py
```
Download the PlantVillage dataset from the Kaggle link above; update `DATA_DIR` in the script to point to the `color` image folder.

## Files
- `plant_disease_detection.py` — full pipeline: data loading/augmentation, transfer learning setup, training loop, evaluation
