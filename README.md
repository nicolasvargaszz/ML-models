

---

# 💻 Machine Learning Models for Complaint Classification
![Python](https://img.shields.io/badge/Python-3.8-blue) ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-0.24-blue) ![TF-IDF](https://img.shields.io/badge/TF--IDF-Algorithm-orange)

This repository focuses on applying **machine learning models** to classify complaint types using natural language processing (NLP) techniques. We aim to predict whether a complaint is related to computer issues or non-computer issues using various ML models.

## 📂 What Kind of Data Do We Have?
We are working with two important `.csv` files that contain:
1. **Computer complaints**
2. **Non-computer complaints**

> All complaints are in **Spanish**, which adds an interesting dimension to the text processing and classification.

![Dataset Overview](https://github.com/nicolasvargaszz/ML-models/assets/65906810/1748a023-1a53-4fce-88d6-2a0f05dbaef0)

As shown above, we are dealing with a moderately sized dataset. 

## 🎯 Goal of the Project
Our goal is to build a **machine learning model** that can accurately classify complaints into two categories:
- **Computer-related complaints**
- **Non-computer-related complaints**

## 🛠️ How Will We Use the Data?
Since machine learning models only work with numerical data, we need to transform our text data (complaints in Spanish) into a numerical format. To achieve this, we will use the **TF-IDF (Term Frequency-Inverse Document Frequency)** algorithm.

## 🔎 What's the TF-IDF Algorithm?
TF-IDF is a popular algorithm in NLP used to convert text into a format that can be processed by machine learning models. It assigns weights to words based on their importance in a document and across a collection of documents.

- **Term Frequency (TF)**: Measures how frequently a term appears in a document.
- **Inverse Document Frequency (IDF)**: Measures how important a term is across the entire collection.

In **Scikit-learn**, this transformation is done using `TfidfVectorizer`.

`from sklearn.feature_extraction.text import TfidfVectorizer`

`legal_tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', ngram_range=(1, 2), stop_words=list(STOP_WORDS))`

`legal_tfidf.fit(df[df['tipo'] == 'denuncia-Legal']['Denuncias'])`

`legal_vocab = legal_tfidf.vocabulary_`

This snippet transforms the complaint texts into numerical vectors and builds a vocabulary from the legal complaint data.

## 📑 Defining the Labels and Features
After applying TF-IDF, we define the features and labels:

`legal_features = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', ngram_range=(1, 2), stop_words=list(STOP_WORDS), vocabulary=legal_vocab).fit_transform(df[df['tipo'] == 'denuncia-Legal']['Denuncias'])`

`legal_labels = labels[df['tipo'] == 'denuncia-Legal']`

The `legal_features` are extracted using the defined vocabulary, and `legal_labels` contain the corresponding labels for the complaints.

![Most Important Words](https://github.com/nicolasvargaszz/ML-models/assets/65906810/104b7019-d11f-46ae-8e87-5648ea3c86c4)

## 🤖 Choosing the Model
We experiment with several machine learning models to find the best one for our classification task:

### RandomForestClassifier

The `RandomForestClassifier` is an ensemble learning method based on decision trees. It creates multiple decision trees and combines their predictions to make the final classification.

- **`n_estimators`**: Number of decision trees.
- **`max_depth`**: Limits the depth of each tree.
- **`random_state`**: Seed for reproducibility.

### LinearSVC

`LinearSVC` (Linear Support Vector Classifier) is a linear model for classification:

- Uses a linear kernel by default.
- Finds the best hyperplane that separates different classes.
- Supports regularization to prevent overfitting.

### MultinomialNB

`MultinomialNB` (Multinomial Naive Bayes) is based on Bayes' theorem:

- Suitable for text classification tasks.
- Models the likelihood of each feature's occurrence given the class.

### LogisticRegression

`LogisticRegression` is a linear model for binary and multi-class classification:

- Estimates the probability of belonging to a certain class using a logistic function.
- Supports regularization to control model complexity.

![Model Comparison](https://github.com/nicolasvargaszz/ML-models/assets/65906810/8a3b087e-acdf-4e64-b6f3-ef9d9a1fdede)

Applying our data to these models, we find that the `LinearSVC` performs the best.

![Model Performance](https://github.com/nicolasvargaszz/ML-models/assets/65906810/d1507b7c-1f6b-46d5-91eb-00cd4b5c3b55)

So, the `LinearSVC` model achieves 100% effectiveness, confirming its superior performance for our dataset.

---


