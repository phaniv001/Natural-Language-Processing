# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 13:51:58 2021

@author: laksh
"""

import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

message = pd.read_csv('D:\Data Science\Datasets\SpamClassifier-master\smsspamcollection\SMSSpamCollection',
                      sep = '\t', names = ['label', 'Message'])

lemmatize = WordNetLemmatizer()
cv = TfidfVectorizer()

corpus = []
for i in range(len(message)):
    review = re.sub('[^a-zA-Z]', ' ', message['Message'][i])
    review = review.lower()
    review = review.split()
    review = [lemmatize.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
X = cv.fit_transform(corpus).toarray()

y = message.iloc[:,:-1]
y = pd.get_dummies(message['label'])
y = y.iloc[:, 1].values

# Split the data into train set and test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.naive_bayes import MultinomialNB

spam_detect_model = MultinomialNB().fit(X_train, y_train)
y_pred = spam_detect_model.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, plot_confusion_matrix
cm = confusion_matrix(y_test, y_pred)

acc_score = accuracy_score(y_test, y_pred)




