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

## Installation

Clone the repository:

```bash
git clone <your-repo-link>
cd <repo-name>
```

Install required packages:

```bash
pip install -r requirements.txt
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

---

## Repository Structure

```
PROJECT_ROOT/
│
├── README.md
├── requirements.txt
│
├── DATA/
│   └── raw/
│
├── SCRIPTS/
│   ├── 01_load_data.py
│   ├── 02_preprocessing.py
│   ├── 03_cnn_model.py
│   ├── 04_vit_knn_model.py
│   └── 05_model_comparison.py
│
├── OUTPUT/
│   ├── cnn_metrics.csv
│   ├── vit_knn_metrics.csv
│   └── confusion_matrices/
│
├── CNN.ipynb
└── VIT.KNN.ipynb
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

From the project root directory:

```bash
pip install -r requirements.txt
```

If using Google Colab:

```python
!pip install datasets transformers scikit-learn torch torchvision matplotlib
```

If errors occur, restart the runtime after installation.

---

### Step 3 — Run the CNN Model

Open:

```
CNN.ipynb
```

Run all cells in order. This will:

- Resize and normalize images  
- Split data into training and testing sets (80/20)  
- Train a CNN model  
- Generate predictions  
- Output:
  - Accuracy  
  - F1-score  
  - Precision  
  - Recall  
- Display confusion matrices and training curves  

---

### Step 4 — Run the ViT + KNN Model

Open:

```
VIT.KNN.ipynb
```

Run all cells in order. This will:

- Load the same dataset  
- Preprocess images using a ViT image processor  
- Convert images into embeddings using a pretrained Vision Transformer  
- Split data into training and testing sets (80/20)  
- Train a KNN classifier on embeddings  
- Generate predictions  
- Output:
  - Accuracy  
  - F1-score  
  - Precision  
  - Recall  
  - PR-AUC  
- Display precision-recall curves and classification reports  

---

### Step 5 — Compare Results

Compare outputs from both models:

- CNN results in `CNN.ipynb`  
- ViT + KNN results in `VIT.KNN.ipynb`  

Key metrics:

- Weighted F1-score (primary metric)  
- Accuracy  
- Precision and Recall  
- Confusion matrices  

---

### Step 6 — Expected Output

Typical results:

- CNN:
  - Accuracy ≈ 0.80+  
  - F1-score ≈ 0.80+  

- ViT + KNN:
  - Accuracy ≈ 0.79  
  - F1-score ≈ 0.79  
  - PR-AUC ≈ 0.77  

---

### Notes

- Dataset is relatively balanced (~53% AI, ~47% real)  
- Many images are blurry, making classification more difficult  
- CNN slightly outperforms ViT + KNN  
- ViT struggles more with identifying real images  
- GPU is recommended but not required  

---

## Modeling Approach

- Image resizing and normalization  
- 80/20 stratified train-test split  
- CNN for image classification  
- ViT used for feature extraction (embeddings)  
- KNN classifier applied to ViT embeddings  
- Primary evaluation metric: Weighted F1 score  

---

## Results

| Model        | F1 Score | Accuracy |
|-------------|---------|----------|
| CNN         | ~0.80+  | ~0.80    |
| ViT + KNN   | 0.79    | 0.79     |

Additional metrics (ViT + KNN):

- Precision: ~0.79  
- Recall: ~0.79  
- PR-AUC: 0.77  

---

## Future Work

- Fine-tune the ViT model instead of embeddings  
- Try different classifiers (Logistic Regression, SVM)  
- Improve image preprocessing  
- Increase dataset size  
- Explore deeper CNN architectures  

---

## Acknowledgements

Dataset: Hugging Face — AI-Generated vs Real Images  
https://huggingface.co/datasets/Hemg/AI-Generated-vs-Real-Images-Datasets  

Instructor: Karsten Siller  
TA: Cole Whittington  
