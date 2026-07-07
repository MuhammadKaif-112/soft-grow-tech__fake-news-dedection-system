# 🚨 Fake News Detection System using NLP & Machine Learning

A machine learning system that detects whether a news article is **fake or real** using Natural Language Processing techniques. Built as the final project of my Machine Learning internship at **SoftGrowthTech**.

---

## 📌 Project Overview

This project implements a complete NLP-based classification pipeline:
- Load and merge a large-scale fake/real news dataset (44,898 articles)
- Explore and analyze language patterns between fake and real news
- Clean and preprocess raw text data using NLP techniques
- Extract features using TF-IDF Vectorization
- Train a Logistic Regression classifier
- Evaluate model performance using standard classification metrics
- Visualize vocabulary differences via Word Cloud
- Predict whether any news article is fake or real with a confidence score

---

## 📊 Dataset

**Source:** [Fake and Real News Dataset (Kaggle)](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

The dataset consists of two CSV files:

| File | Description | Articles |
|---|---|---|
| Fake.csv | Articles from unreliable/flagged sources | 23,481 |
| True.csv | Articles from Reuters.com | 21,417 |
| **Total** | **Combined dataset** | **44,898** |

Each article contains: `title`, `text`, `subject`, `date`

**Class distribution:** 52.3% Fake / 47.7% Real — nearly perfectly balanced, no class imbalance issues.

---

## 🛠️ Tech Stack

- **Python**
- **Pandas** — data loading & manipulation
- **Scikit-learn** — TF-IDF vectorization, model training & evaluation
- **Matplotlib / Seaborn** — data visualization
- **WordCloud** — vocabulary visualization
- **Re / String** — text cleaning

---

## 🔄 Workflow

### 1. Data Loading & Merging
Loaded both `Fake.csv` and `True.csv`, assigned labels (0=Fake, 1=Real), and merged into a single 44,898-row DataFrame.

### 2. Exploratory Data Analysis (EDA)
- Confirmed balanced class distribution (52.3% fake, 47.7% real)
- Analyzed article length: fake articles average 423 words vs 385 for real articles
- Dropped `subject` column — subjects were completely different between fake/real articles, making it a "cheat" feature that would prevent the model from learning genuine language patterns

### 3. Text Preprocessing & Cleaning
Combined `title` + `text` into a single `content` column, then applied:
- Lowercasing
- URL removal
- Punctuation removal
- Newline removal
- Removal of words containing numbers
- Extra whitespace removal

### 4. TF-IDF Feature Extraction
- Split data first (80/20) to prevent data leakage
- Applied `TfidfVectorizer` with `max_features=50,000` and `ngram_range=(1,2)` — capturing both single words and two-word phrases like "white house", "donald trump"
- Fit vectorizer on training data only, then transformed both train and test sets

### 5. Model Training
Trained a **Logistic Regression** classifier on 35,918 articles across 50,000 TF-IDF features.

### 6. Model Evaluation

| Metric | Value |
|---|---|
| **Accuracy** | **98.85%** |
| Precision | 0.9868 |
| Recall | 0.9889 |
| F1 Score | 0.9879 |

**Confusion Matrix:**

|  | Predicted: Fake | Predicted: Real |
|---|---|---|
| **Actual: Fake** | 4,677 | 56 |
| **Actual: Real** | 47 | 4,200 |

Out of 8,980 test articles, only **103 mistakes total**.

### 7. Word Cloud Visualization
Generated side-by-side word clouds revealing clear vocabulary differences:
- **Fake news** uses emotional, conversational language: "people", "think", "know", "believe", "claim", "attack"
- **Real news** uses formal, institutional language: "reuters", "white house", "washington", "percent", "congress", "thursday"

This visual confirms the model is learning genuine linguistic patterns, not random noise.

### 8. Live Predictor
Built a `predict_news()` function that takes any news text and returns:
- Prediction: FAKE NEWS or REAL NEWS
- Confidence score (%)
- Full fake/real probability breakdown

**Example output:**
```
==================================================
  PREDICTION  : 🚨 FAKE NEWS
  CONFIDENCE  : 73.30%
  Fake prob   : 73.30%
  Real prob   : 26.70%
==================================================
```

---

## 📈 Results & Insights

The model achieved **98.85% accuracy** on 8,980 unseen test articles — correctly identifying fake and real news in all but 103 cases. Both precision (0.987) and recall (0.989) are exceptionally high, meaning the model is reliable in both directions: it rarely raises false alarms and rarely misses actual fake news.

The word cloud analysis revealed that fake news relies heavily on emotional, opinion-driven vocabulary ("think", "believe", "don't", "people") while real news consistently uses formal, source-attributed language ("reuters", "said", "percent", "white house"). This linguistic difference is what gives the TF-IDF + Logistic Regression pipeline its strong discriminative power.

**Possible next steps:** experiment with deep learning approaches (LSTM, BERT) for even higher accuracy, or extend the system to detect fake news in other languages.

---

## 📁 Repository Structure

```
fake-news-detection/
├── fake_news_detection.ipynb        # Main notebook (full pipeline)
├── Fake.csv                          # Fake news articles
├── True.csv                          # Real news articles
├── wordcloud_comparison.png          # Fake vs Real vocabulary visualization
├── class_distribution.png           # Class balance chart
├── length_distribution.png          # Article length distribution
└── README.md
```

## 🚀 How to Run

1. Clone this repository
2. Install dependencies:
```bash
pip install pandas scikit-learn matplotlib seaborn wordcloud
```
3. Open `fake_news_detection.ipynb` in Jupyter Notebook
4. Run all cells in order
5. Use `predict_news("paste any article text here")` to test live predictions

---

## 🎓 Internship

This project was completed as the **final project** of my Machine Learning internship at **SoftGrowthTech**.

---
**Author:** Muhammad Kaif
**GitHub:** [MuhammadKaif-112](https://github.com/MuhammadKaif-112)
