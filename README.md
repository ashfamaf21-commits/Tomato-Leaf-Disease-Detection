# Tomato Leaf Disease Detection with Deep Learning and Grad-CAM

A deep learning model that classifies tomato leaf photos as healthy or one of nine common diseases, with **Grad-CAM** used to visualize which regions of the leaf the model relies on for its predictions.

## Overview

This project applies transfer learning to tomato leaf disease classification and layers on Grad-CAM explainability, so the model's predictions can be visually inspected rather than treated as a black box.

## Dataset

- **Source:** [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) (Abdallah Ali, Kaggle)
- **Subset used:** Tomato leaves only — 10 classes (healthy + 9 diseases), ~18,000 images
- **Split:** Stratified 70% train / 15% validation / 15% test, `seed=42`

## Method

- **Architecture:** Transfer learning with **MobileNetV2** (ImageNet-pretrained, frozen base) + a new classification head trained on the tomato classes
- **Preprocessing:** Images resized to 224×224, rescaled to MobileNetV2's expected input range
- **Training:** Data augmentation on the training set, early stopping on validation accuracy, best checkpoint retained
- **Evaluation:** Held-out test set (15%, untouched until final evaluation)

## Results

| Metric | Score |
|---|---|
| Test accuracy | **89.0%** |
| Macro F1-score | **0.854** |
| Weighted F1-score | **0.890** |

**Best-performing classes:**
- Tomato Yellow Leaf Curl Virus — F1 0.977 (precision 0.995, recall 0.959)
- Healthy — F1 0.971 (precision 0.963, recall 0.979)
- Late blight — F1 0.923 (precision 0.952, recall 0.895)

**Weakest-performing classes:**
- Early blight — F1 0.669 (recall only 0.547 — the model misses more than half of true Early blight cases)
- Target Spot — F1 0.746
- Spider mites (Two-spotted) — F1 0.828 (precision 0.748 — frequently over-predicted)

Early blight's low recall alongside Target Spot's moderate precision suggests these two classes (and Spider mites) are visually similar and are the most-confused pair(s) in the confusion matrix — worth a closer look if extending this project.

Full per-class metrics are in [`classification_report.csv`](classification_report.csv).

## Explainability (Grad-CAM)

Grad-CAM heatmaps were generated for test images to visualize which leaf regions the model attended to when making its prediction. *(Fill in: did the model focus on the actual lesion areas? Any cases where it focused on background/edges instead — e.g., for the weaker classes like Early blight?)*

## Limitations

- PlantVillage images were captured in controlled lab conditions on plain backgrounds — real field photos (variable lighting, cluttered backgrounds, multiple overlapping leaves) would likely yield lower accuracy
- Single crop (tomato) only — not validated on other species
- No field validation was performed; all evaluation is on held-out lab-condition images

## Future Work

- Evaluate on real field-captured images
- Extend to other crops (e.g., rice, tea)
- Compare against other backbones (e.g., EfficientNet)
- Investigate the Early blight / Target Spot / Spider mites confusion with targeted augmentation or additional training data

## Repository Contents

- `Tomato_Leaf_Disease_GradCAM.ipynb` — full training and evaluation notebook
- `classification_report.csv` — per-class precision, recall, F1-score on the test set
- `tomato_disease_mobilenetv2.keras` — trained model weights
