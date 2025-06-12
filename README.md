# sentiment_analysis
# 🧠 Multi-Class Sentiment Analysis with Transformers

This project demonstrates how to build a **multi-class sentiment analysis model** using a large dataset of over **241,000 English-language comments**, classified into three categories:

- **0** — Negative  
- **1** — Neutral  
- **2** — Positive  

The model is fine-tuned using **DistilBERT** from Hugging Face Transformers, optimized for performance on Google Colab with GPU support.

---

## 📁 Dataset

The dataset contains preprocessed English comments from kaggle sources

Each sample includes:
- `text`: Cleaned comment
- `label`: Sentiment class (0 = Negative, 1 = Neutral, 2 = Positive)

---

## 🧰 Tech Stack

- [Transformers (Hugging Face)](https://huggingface.co/transformers/)
- [Datasets (Hugging Face)](https://huggingface.co/docs/datasets)
- [PyTorch](https://pytorch.org/)
- [scikit-learn](https://scikit-learn.org/)
- [Google Colab](https://colab.research.google.com/) (for training)

---

📊 Results (Subset: 10K Train / 2K Test)
Metric	Value
Accuracy	78.35%
F1 Score	0.782
Precision	0.783
Recall	0.783
Val Loss	0.598


💡 Future Enhancements

    🏋️‍♀️ Train on the full 241K dataset

    🌐 Build a live sentiment prediction app using Gradio or Streamlit

    🔬 Experiment with RoBERTa and BERT for performance comparison

    📈 Visualize confusion matrix and class-level performance

## 🚀 Quick Start

```bash
# Install dependencies
pip install transformers datasets scikit-learn pandas torch


