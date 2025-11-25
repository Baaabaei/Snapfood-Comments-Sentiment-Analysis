# Snapfood Comments Sentiment Analysis

A comparative study of three machine learning approaches for Persian sentiment analysis on Snapfood restaurant reviews.

## 📊 Dataset

- **Source**: Snapfood Persian comments dataset
- **Dataset Location**: Available on Kaggle
- **Model Info**: [ParsBERT on Hugging Face](https://huggingface.co/HooshvareLab/bert-base-parsbert-uncased)
- **Size**: 52,110 training samples, ~9,000 test samples
- **Classes**: Binary classification (HAPPY=0, SAD=1)
- **Balance**: ~50% for each class

## 🔍 Models Implemented

### 1. TF-IDF + Logistic Regression
- **Preprocessing**: hazm library for Persian stopwords removal
- **Vectorization**: TF-IDF with custom preprocessing pipeline
- **Model**: Logistic Regression (max_iter=1000)
- **Results**:
  - Accuracy: 81.92%
  - Precision: 80.18%
  - Recall: 85.13%

### 2. RNN (LSTM)
- **Architecture**: 
  - Embedding layer (10,000 words, 50 dimensions)
  - LSTM layer (32 units)
  - Dense layer (32 units, ReLU)
  - Output layer (1 unit, sigmoid)
- **Training**: 10 epochs, batch size 32
- **Results**:
  - Accuracy: 81.77%
  - Precision: 81.73%
  - Recall: 82.14%

### 3. ParsBERT (Pre-trained Transformer)
- **Model**: HooshvareLab/bert-base-parsbert-uncased
- **Fine-tuning**: 3 epochs, batch size 16
- **Best Performance**:
  - Accuracy: **86.94%** ✨
  - Precision: 85.57%
  - Recall: 89.07%

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy scikit-learn
pip install tensorflow keras
pip install transformers torch
pip install hazm  # Persian NLP library
```

### Running the Notebook

1. Clone the repository
2. Download the Snapfood dataset from [Kaggle](https://www.kaggle.com/)
3. Update file paths in the notebook
4. Run cells sequentially

## 📈 Model Comparison

| Model | Accuracy | Precision | Recall | Training Time |
|-------|----------|-----------|--------|---------------|
| TF-IDF + LR | 81.92% | 80.18% | 85.13% | Fast |
| LSTM | 81.77% | 81.73% | 82.14% | Moderate |
| **ParsBERT** | **86.94%** | **85.57%** | **89.07%** | Slow |

## 🔧 Key Features

- **Persian Language Support**: Proper handling of Persian text with hazm library
- **Multiple Approaches**: Classical ML, Deep Learning, and Transfer Learning
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, and Confusion Matrix
- **Visualization**: Confusion matrices with seaborn

## 📝 Code Structure

```
├── Data Loading & EDA
├── Preprocessing (stopwords, tokenization)
├── Model 1: TF-IDF + Logistic Regression
├── Model 2: RNN (LSTM)
└── Model 3: ParsBERT Fine-tuning
```

## 🎯 Conclusions

- ParsBERT significantly outperforms classical and basic deep learning approaches
- Transfer learning with pre-trained Persian models is highly effective
- TF-IDF baseline provides surprisingly competitive results
- LSTM shows balanced performance between traditional and modern approaches

## 📚 References

- [ParsBERT](https://github.com/hooshvare/parsbert)
- [Hazm - Persian NLP](https://github.com/sobhe/hazm)
- [Transformers by Hugging Face](https://huggingface.co/transformers/)

## 👤 Author

[Your Name]

## 📄 License

This project is open source and available under the MIT License.

---

**Note**: This notebook was originally run on Kaggle with GPU acceleration for faster training of deep learning models.
