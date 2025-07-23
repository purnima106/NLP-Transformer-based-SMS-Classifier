# SMS Spam Detection (NLP Project)

A simple **Natural Language Processing (NLP)** project for detecting spam messages using both **classical ML techniques (TF-IDF + Naive Bayes/Logistic Regression)** and **modern Transformer embeddings (DistilBERT)**.

This project is designed to learn and implement core NLP terminologies and techniques like **text preprocessing, tokenization, stopword removal, Bag-of-Words (BoW), TF-IDF, embeddings, and classification.**

---

## **Project Overview**
- **Goal:** Classify SMS messages as `spam` or `ham` (not spam).
- **Dataset:** [SMS Spam Collection (UCI ML Repository)](https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip)
- **Approach:**
  - Data cleaning (lowercasing, punctuation removal, etc.).
  - Tokenization and stopword removal using **NLTK**.
  - Feature extraction via **BoW** and **TF-IDF**.
  - Model training using **Multinomial Naive Bayes** and **Logistic Regression**.
  - Advanced embeddings using **DistilBERT** with Logistic Regression.
  - Evaluation with metrics like Accuracy, Precision, Recall, and F1-score.

---

## **Technologies Used**
- **Language:** Python 3
- **Libraries:** `pandas`, `scikit-learn`, `nltk`, `transformers`, `torch`
- **Tools:** Jupyter Notebook, GitHub

---

## **Setup Instructions**
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/nlp-sms-spam.git
   cd nlp-sms-spam
Create and activate a virtual environment (optional but recommended):

bash
Copy code
python -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate     # For Windows
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Download the dataset and place it inside the data/ folder:
SMS Spam Dataset

Launch Jupyter Notebook:

bash
Copy code
jupyter notebook
Project Status
Current Progress:

Data loading and cleaning ✅

Tokenization & stopword removal ✅

TF-IDF + Classical ML models ✅

DistilBERT embeddings 🚧 (in progress)

Evaluation and visualization 🚧 (upcoming)

Future Work
Add Named Entity Recognition (NER) and sentiment analysis pipelines.

Hyperparameter tuning for Logistic Regression.

Experiment with other transformer models like BERT or RoBERTa.

How to Contribute
Pull requests and suggestions are welcome! Please open an issue if you find any bug or have improvement ideas.

License
This project is licensed under the MIT License.

# Author: Purnima Nahata
