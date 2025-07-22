# Fake-news-detection-using-PCA-and-genetic-algorithm

This project implements a robust and efficient fake news detection system that combines contextual language models, dimensionality reduction, and evolutionary feature selection for high accuracy and interpretability.

## Overview

The objective is to classify news articles as **real** or **fake** using a hybrid approach:
- **Contextual Embeddings** from DistilBERT
- **Dimensionality Reduction** with PCA
- **Feature Selection** using Genetic Algorithm (GA)
- **Final Classification** using Logistic Regression

## Motivation

Fake news poses serious threats to society. Traditional models struggle with high-dimensional and noisy text data. Our solution aims to:
- Improve semantic understanding with BERT-based embeddings
- Reduce feature space with PCA
- Select optimal features using Genetic Algorithm
- Achieve high precision with Logistic Regression

---

## Tech Stack

| Component         | Tool / Library          |
|------------------|-------------------------|
| Language Model    | `transformers (DistilBERT)` |
| Dimensionality Reduction | `scikit-learn (PCA)`       |
| Feature Selection | `deap` (Genetic Algorithm) |
| Classifier        | `scikit-learn (LogisticRegression)` |
| Evaluation        | `Precision`, `Recall`, `F1-score`, `Accuracy` |
| Platform          | `Google Colab` (No GPU) |

---

## Dataset

- 500 **Fake** news + 500 **Real** news articles
- Combined from standard sources like `fake.csv` and `true.csv`
- Balanced to reduce memory consumption

---

## Model Pipeline

1. **Data Preprocessing**
   - Lowercasing, cleaning, tokenization
2. **Text Embedding**
   - DistilBERT converts text to 768-d vectors
3. **PCA**
   - Reduces vectors to ~100 dimensions
4. **Genetic Algorithm**
   - Evolves to select optimal subset of features
5. **Logistic Regression**
   - Trained on selected features

---

## Results

| Metric     | Value (example) |
|------------|-----------------|
| Accuracy   | 92.5%           |
| Precision  | 93.2%           |
| Recall     | 91.4%           |
| F1-Score   | 92.3%           |

> *Note: These are representative values; actual results depend on dataset split and hyperparameters.*

---

## How to Run

1. Clone the repo:
```bash
git clone https://github.com/YourUsername/FakeNews-Detection-PCA-GA.git
cd FakeNews-Detection-PCA-GA
