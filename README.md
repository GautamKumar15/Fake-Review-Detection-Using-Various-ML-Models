# 🕵️ Fake Review Detection Using Various ML Models

This project focuses on detecting fake or deceptive product reviews using multiple machine learning algorithms. It provides a practical pipeline including data preprocessing, feature engineering, model training, and evaluation—all demonstrated through two structured Jupyter notebooks.

---

## 📁 Project Structure
Fake-Reviews-Detection-main/
├── Fake Reviews Detection Part-1.ipynb # Preprocessing, EDA, TF-IDF
├── Fake Reviews Detection Part-2 Continued.ipynb # Model training & evaluation
├── fake reviews dataset.csv # Original dataset
├── Preprocessed Fake Reviews Detection Dataset.csv # Cleaned dataset
└── README.md

---

## 📌 Objectives

- Detect fake product reviews using ML classification.
- Preprocess text data (cleaning, tokenizing, stopword removal).
- Compare multiple ML models like Logistic Regression, SVM, Random Forest.
- Evaluate using metrics like Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.

---

## 🧰 Tools & Libraries Used

- **Languages:** Python
- **IDE:** Jupyter Notebook
- **Libraries:**  
  - `pandas`, `numpy` – Data handling  
  - `nltk`, `re` – Text cleaning  
  - `scikit-learn` – ML modeling and evaluation  
  - `matplotlib`, `seaborn` – Visualization  

---

## 📊 ML Models Used

- Logistic Regression
- Naive Bayes
- Support Vector Machine (SVM)
- Random Forest Classifier

---

## 🔍 Workflow Summary

1. **Part-1 Notebook:**
   - Load and explore the dataset
   - Clean text (lowercase, remove punctuation, stopwords)
   - Preprocess using TF-IDF
   - Visualizations (label distribution, word frequency)

2. **Part-2 Notebook:**
   - Train-test split
   - Apply ML models
   - Evaluate using classification metrics
   - Compare model performance

---

## 🧪 Sample Evaluation Metrics

| Model               | Accuracy | Precision | Recall | F1 Score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression| ~88%     | High      | High   | Good     |
| Naive Bayes        | ~85%     | Medium    | High   | Fair     |
| SVM                | ~89%     | High      | High   | Good     |
| Random Forest      | ~91%     | High      | High   | Best     |

> *Actual results may vary based on preprocessing and data split.*

---

## 📈 Future Improvements

- Add Deep Learning models (LSTM, BERT).
- Use Word Embeddings instead of TF-IDF.
- Deploy as a web app (Streamlit or Flask).
- Extend to multilingual review datasets.

---

## 🙌 Author

**Gautam Kumar**  
Pre-final year B.Tech in AI & Data Science  
Email: *gk7088202@gmail.com* (optional)  


---



