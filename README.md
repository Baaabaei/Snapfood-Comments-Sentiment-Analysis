# Snapfood Comments Sentiment Analysis

[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://github.com/Baaabaei/Snapfood-Comments-Sentiment-Analysis)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive comparative study of three distinct machine learning approaches for Persian sentiment analysis on real-world restaurant reviews from the Snapfood platform.

## 📋 Overview

This project benchmarks three different methodologies—ranging from classical machine learning to modern deep learning—to classify Persian comments about food and restaurant experiences. It demonstrates how transfer learning with a pre-trained Persian BERT model significantly outperforms other approaches.

**Key Research Questions:**
- How well do classical ML methods perform on Persian text?
- Can a simple LSTM capture semantic meaning effectively?
- Does a pre-trained transformer model (ParsBERT) provide superior results for this low-resource language task?

## 📊 Dataset

The dataset consists of user-submitted comments and ratings from the Snapfood delivery platform.

| Feature | Description |
| :--- | :--- |
| **Source** | Snapfood (Iranian food delivery platform) |
| **Availability** | Downloadable from [Kaggle](https://www.kaggle.com/) (Search: "Snapfood Comments") |
| **Total Samples** | ~61,000 |
| **Split** | 52,110 Training / ~9,000 Test |
| **Classes** | Binary (HAPPY = 0, SAD = 1) |
| **Balance** | Approximately 50% for each class |

> **Note:** The dataset is not included in this repository. You must download it separately from Kaggle and update the file paths in the notebook.

## 🧠 Models Implemented

The project implements and compares three models, from simple to complex:

### 1. TF-IDF + Logistic Regression
A classical Natural Language Processing (NLP) pipeline.

- **Preprocessing:** Persian stopword removal using the `hazm` library.
- **Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency).
- **Classifier:** Logistic Regression.
- **Result:** Provides a strong, fast, and interpretable baseline.

### 2. RNN (LSTM)
A simple Recurrent Neural Network using an LSTM (Long Short-Term Memory) layer to capture sequence information.

- **Architecture:** Embedding Layer → LSTM (32 units) → Dense (32, ReLU) → Output (Sigmoid).
- **Training:** 10 epochs with a batch size of 32.
- **Result:** Balanced performance, capturing more context than TF-IDF but limited by data size.

### 3. ParsBERT (Pre-trained Transformer)
A fine-tuned version of the state-of-the-art ParsBERT model, specifically pre-trained on a large Persian corpus.

- **Model:** `HooshvareLab/bert-base-parsbert-uncased` from Hugging Face.
- **Fine-tuning:** 3 epochs with a batch size of 16.
- **Result:** Achieves the best performance, demonstrating the power of transfer learning.

## 🚀 Getting Started

Follow these instructions to run the project locally.

### Prerequisites

Create a virtual environment and install the required libraries:

```bash
pip install pandas numpy scikit-learn
pip install tensorflow keras
pip install transformers torch
pip install hazm  # For Persian text processing
