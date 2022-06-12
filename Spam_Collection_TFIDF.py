# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 22:55:30 2021

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
data = message.copy()
    
#### **************************** Cleaning the data ************************************* ####
lemmatize = WordNetLemmatizer()
cv = TfidfVectorizer()
#cv = CountVectorizer()
corpus = []
for i in range(len(message)):
    review = re.sub('[^a-zA-Z]', ' ', message['Message'][i])
    review = review.lower()
    review = review.split()
    review = [lemmatize.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

#y_train = pd.get_dummies(strat_train['label'])
#y_train = y_train.iloc[:, 1].values
### *************************** Cleaning data and Preparing the Test set ********************************* ####
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits = 20, test_size = 0.2, random_state = 0)
for train_index, test_index in split.split(message, message['label']):
    strat_train = message.loc[train_index]
    strat_test = message.loc[test_index]

X_train = cv.fit_transform(strat_train['Message']).toarray()
y_train = pd.get_dummies(strat_train['label'])
y_train = y_train.iloc[:, 1].values


#X_test = cv.fit_transform(corpus_test).toarray()
#y_test = pd.get_dummies(strat_test['label'], drop_first = True).values
#y_test = pd.get_dummies(strat_test['label'])
#y_test = y_test.iloc[:, 1].values
#### ********* Craete word to vectors using TF-IDF ******************** ####

#from sklearn.naive_bayes import MultinomialNB
#from sklearn.metrics import confusion_matrix, accuracy_score
#spam_detect_model = MultinomialNB().fit(X_train, y_train)
#y_pred = spam_detect_model.predict(X_test)
#cm = confusion_matrix(y_test, y_pred)
#print(accuracy_score(y_test, y_pred))



    
