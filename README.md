# Hate-Speech-Detection-
Overview
This project aims to detect hate speech in tweets using a machine learning model. The dataset contains tweets labeled as "Hatespeech", "Offensive Language", or "Neither". The project involves data preprocessing, feature extraction, model training, and evaluation.

Table of Contents
1)Installation
2)Dataset
3)Data Preprocessing
4)Model Training and Evaluation
5)Results
6)Usage
7)Contributing

Installation
Clone the repository:
git clone

Dataset
The dataset used in this project is twitter_data.csv, which includes the following columns:

tweet: The text of the tweet.
class: The class label (0 for "Hatespeech", 1 for "Offensive Language", 2 for "Neither").
Data Preprocessing
The data preprocessing steps include:

Removing URLs, HTML tags, and punctuation.
Converting text to lowercase.
Removing stopwords.
Stemming the text.
Model Training and Evaluation
Feature Extraction: Using CountVectorizer to convert the text data into a matrix of token counts.
Model: Training a Decision Tree classifier.
Evaluation: Using a confusion matrix and accuracy score to evaluate the model's performance.

Results
The model achieved an accuracy of approximately 87.61% on the test data.

Usage
Preprocess the data:

import pandas as pd
from preprocess import clean_data
dataset = pd.read_csv("twitter_data.csv")
dataset["labels"] = dataset["class"].map({0: "Hatespeech", 1: "Offensive Language", 2: "Neither"})
data = dataset[["tweet", "labels"]]
data["tweet"] = data["tweet"].apply(clean_data)
Train the model:

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X = data["tweet"]
Y = data["labels"]

cv = CountVectorizer()
X = cv.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

dt = DecisionTreeClassifier()
dt.fit(X_train, Y_train)

Evaluate the model:
from sklearn.metrics import confusion_matrix, accuracy_score

Y_pred = dt.predict(X_test)
cm = confusion_matrix(Y_test, Y_pred)
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Confusion Matrix:\n{cm}")
print(f"Accuracy: {accuracy}")

Visualize the results:
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(cm, annot=True, fmt=".1f", cmap="YlGnBu")
plt.show()

Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.


