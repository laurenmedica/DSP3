# Identifying AI-Generated vs. Human-Made Images

Compare whether AI-generated images can be detected using a CNN model versus a ViT + KNN model using image data.

**Course:** DS 4002  
**Group Name:** DCL  
**Group Leader:** Dev Patel  
**Group Members:** Lauren Medica, Dev Patel, Caroline Lingle  

---

## Overview

This project compares two machine learning approaches:

- Convolutional Neural Network (CNN) classification  
- Vision Transformer (ViT) + K-Nearest Neighbors (KNN) classification  

We evaluate performance primarily using weighted F1 score.

**Key Result:**  
The ViT + KNN model achieved an F1-score of approximately **0.79**, approaching the target goal of 0.80. The CNN model performed slightly better overall.

---
```

---

## Software & Platform

**Language:** Python 3  

**Libraries Used:**
- numpy  
- pandas  
- scikit-learn  
- torch  
- torchvision  
- transformers  
- datasets  
- matplotlib
- tensorflow
- keras  

---

## Repository Structure

```
```
PROJECT_ROOT/
│
├── README.md
│
├── DATA/
│   └── raw/
│
├── OUTPUT/
│   ├── output_1.pdf
│   ├── output_2.pdf
│   ├── output_3.pdf
│   ├── output_4.pdf
│   └── output_5.pdf
│
├── SCRIPTS/
│   ├── P3EDATake2.ipynb
│   └── P3CNN_VIT.ipynb
```

```

---

## Reproducing Results

### Step 1 — Load Data

This project uses the Hugging Face dataset:

```python
from datasets import load_dataset
dataset = load_dataset("Hemg/AI-Generated-vs-Real-Images-Datasets")
```

The dataset contains labeled images classified as:
- AI-generated (AiArtData)  
- Human-made (RealArt)  

---

### Step 2 — Install Dependencies

Install required libraries manually:

```python
!pip install datasets transformers scikit-learn torch torchvision matplotlib
```

Restart the runtime if needed.

---

### Step 3 — Run Combined Model File

Open:

```
P3CNN_VIT.ipynb
```

This notebook includes both approaches in one file and allows for direct comparison between models.

---

### Step 4 — Expected Output

The `OUTPUT/` folder should contain **5 PDF files**, including:

- Model performance summaries  
- Confusion matrices  
- Training curves  
- Precision-recall visualizations  

Typical results:

- CNN:
  - Accuracy ≈ 0.80+  
  - F1-score ≈ 0.80+  

- ViT + KNN:
  - Accuracy ≈ 0.79  
  - F1-score ≈ 0.79  
  - PR-AUC ≈ 0.77  

---

## Modeling Approach

- Image resizing and normalization  
- 80/20 train-test split  
- CNN for image classification  
- ViT used for feature extraction  
- KNN classifier applied to embeddings  
- Primary evaluation metric: Weighted F1 score  

---

## Results

| Model        | F1 Score | Accuracy |
|-------------|---------|----------|
| CNN         | ~0.80+  | ~0.80    |
| ViT + KNN   | 0.79    | 0.79     |

---

## Future Work

- Fine-tune the Vision Transformer  
- Test additional classifiers (SVM, Logistic Regression)  
- Improve preprocessing techniques  
- Expand dataset size  
- Explore deeper CNN architectures  

---

## Acknowledgements

Dataset: Hugging Face — AI-Generated vs Real Images  
https://huggingface.co/datasets/Hemg/AI-Generated-vs-Real-Images-Datasets  

Instructor: Karsten Siller  
TA: Cole Whittington  
