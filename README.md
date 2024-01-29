# ML-models
models that we used to analize the data we have

# What kind of Data we have?

We have two importan .csv files and they contain computer complaints, and the other one normal complaints (All this complains are in spanish)


# Whats our goal with that data.

So we want to train a ML model with our data, and we're going to develop a model that predic the clasification of our complain.
We just hace two option: computer-complain and no-computer-complain.
 
![denuncias](https://github.com/nicolasvargaszz/ML-models/assets/65906810/1748a023-1a53-4fce-88d6-2a0f05dbaef0)
That's the size of our data.

as you can see, we don't have a massive cuantity of data.

# How will we use our that.

It's well know that the ML models only understand numbers, and not str.
Since we have our complains in str we need to change that, and for doing that, we're going to use the TF-idf algorithm.

## What's the TF-idf algorithm 
Certainly! TF-IDF stands for Term Frequency-Inverse Document Frequency. It is a popular algorithm used in information retrieval and text mining to measure the importance of a term in a document within a collection of documents.
fun fact: google use this algorith in the search bar.
Term Frequency (TF) is a measure of how frequently a term appears in a document. It is calculated by dividing the number of times a term appears in a document by the total number of terms in the document. The idea behind TF is that the more times a term appears in a document, the more important it is to that document.

Inverse Document Frequency (IDF) is a measure of how important a term is across the entire collection of documents. It is calculated by taking the logarithm of the ratio of the total number of documents to the number of documents that contain the term. The IDF value decreases as the term appears in more documents, thus giving higher weight to terms that are rare and unique.

The TF-IDF score for a term in a document is obtained by multiplying the TF value with the IDF value. The higher the TF-IDF score, the more relevant the term is to the document.

In scikit-learn, you can use the TfidfVectorizer

by now you should know that we can't apply all this calculus to letter, so thats why we use the TfidfVectorizer, to convert or complains in a matrix.

# Defining the Labels and features for the TF-idf algorithm 

legal_tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', ngram_range=(1, 2), stop_words=list(STOP_WORDS))
legal_tfidf.fit(df[df['tipo'] == 'denuncia-Legal']['Denuncias'])
legal_vocab = legal_tfidf.vocabulary_

here we use the TfidfVectorizer for out vocabulary and then we'll define our labels and feature

legal_features = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', ngram_range=(1, 2), stop_words=list(STOP_WORDS), vocabulary=legal_vocab).fit_transform(df[df['tipo'] == 'denuncia-Legal']['Denuncias'])
legal_labels = labels[df['tipo'] == 'denuncia-Legal']

And with a lit more we will have the most importan words: 
![image](https://github.com/nicolasvargaszz/ML-models/assets/65906810/104b7019-d11f-46ae-8e87-5648ea3c86c4)


Now come the most important part of the project:

# Choosing the model we'll use.

RandomForestClassifier:

# The RandomForestClassifier 
is an ensemble learning method based on decision trees. It creates multiple decision trees and combines their predictions to make the final classification.
The "n_estimators" parameter specifies the number of decision trees to be created in the random forest. A higher number can potentially improve the model's performance but may also increase training time.
The "max_depth" parameter limits the depth of each decision tree in the random forest. It helps control overfitting by restricting the complexity of the trees.
The "random_state" parameter is used to set a seed value for random number generation, ensuring reproducibility of results.
LinearSVC:

# LinearSVC
(Linear Support Vector Classifier) is a linear model for classification. It applies the principles of Support Vector Machines (SVM) to solve binary and multi-class classification problems.
Unlike traditional SVM, LinearSVC uses a linear kernel by default and scales well to large datasets.
It seeks to find the best hyperplane that separates different classes, maximizing the margin between them.
LinearSVC supports the use of regularization techniques, such as L1 or L2 regularization, to control model complexity and prevent overfitting.
MultinomialNB:

# MultinomialNB
(Multinomial Naive Bayes) is a probabilistic classification algorithm based on Bayes' theorem with the assumption of feature independence.
It is commonly used for text classification tasks, such as document categorization or spam filtering.
MultinomialNB models the likelihood of each feature's occurrence given the class and estimates the class probabilities using Bayes' theorem.
It works well with discrete features and is suitable for problems with multiple classes.
LogisticRegression:

# LogisticRegression
is a linear model for binary and multi-class classification.
Despite its name, LogisticRegression is primarily used for classification tasks rather than regression.
It estimates the probability of belonging to a certain class using a logistic function (sigmoid function).
The "random_state" parameter sets a seed value for random number generation, ensuring consistent results.
These models are commonly used in machine learning for classification tasks, and each has its own strengths and weaknesses. It's important to experiment and tune the model parameters based on your specific dataset and problem to achieve optimal performance.

![image](https://github.com/nicolasvargaszz/ML-models/assets/65906810/8a3b087e-acdf-4e64-b6f3-ef9d9a1fdede)

applying our data to that model we get that the best model for us is the linearSVC, but how can we be sure about it.
well people say that numbers don't lie, but let's check.

![image](https://github.com/nicolasvargaszz/ML-models/assets/65906810/d1507b7c-1f6b-46d5-91eb-00cd4b5c3b55)

So there's an excel file in this repo where we can see the same input and diferents outputs from the diferent models, at the end the only one with 100% of efectivity was the linearSVC.
So ones again, we can tell that numbers don't lie.

