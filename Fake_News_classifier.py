# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 12:48:18 2021

@author: laksh
"""

import pandas as pd
import numpy as np
import nltk, re
import matplotlib.pyplot as plt
data = pd.read_csv(r"D:\Data Science\NLP\Fake-News-Classifier-master\Fake-News-Classifier-master\fake-news\test.csv")
data_label = pd.read_csv(r'D:\Data Science\NLP\Fake-News-Classifier-master\Fake-News-Classifier-master\fake-news\submit.csv')
data_label.set_index('id', inplace = True)
data['label'] = data['id'].map(data_label['label'])

#X = data.drop(['label'], axis = 1)   # Independednt features
#y = data['label']   # Dependednt Features

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
data = data.dropna()
message = data.copy()
#index = list(range(0, len(message)))
message = message.reset_index()

### Data Cleaning ###
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
#lemmatize = WordNetLemmatizer()
ps = PorterStemmer()
for i in range(len(message)):
    review = re.sub('[^a-zA-Z]', ' ', message['title'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

#Apply Countvectorizer - Creating Bag o words model

cv = CountVectorizer(max_features = 5000, ngram_range = (1,3))
X = cv.fit_transform(corpus).toarray()
y = message['label']

# To Check the feature names
print("Feature Names :\n", cv.get_feature_names()[:20])
print("\n Params :\n", cv.get_params())
# Split the data to training set and testing set #

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

count_df = pd.DataFrame(X_train, columns = cv.get_feature_names())

## Creating Model
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, plot_confusion_matrix

fake_news_classifier = MultinomialNB().fit(X_train, y_train)
prediction = fake_news_classifier.predict(X_test)
confusion_mat = confusion_matrix(y_test, prediction)
accuracy = accuracy_score(y_test, prediction)
class_report = classification_report(y_test, prediction)
plot_confusion_matrix(fake_news_classifier, X_test, y_test,
                      display_labels=['Fake', 'Real'], cmap=plt.cm.Blues)

print("Accuracy : ", accuracy)
print("\n Classifiaction Report :\n", class_report)

## ************************** Passive Aggressive Model ************************************** # 
from sklearn.linear_model import PassiveAggressiveClassifier

passive_agg_model = PassiveAggressiveClassifier(max_iter=50).fit(X_train, y_train)
pred = passive_agg_model.predict(X_test)
pass_agg_accuracy = accuracy_score(y_test, pred)
plot_confusion_matrix(passive_agg_model, X_test, y_test, display_labels = ['Fake', 'Real'],
                     cmap = plt.cm.Blues)

# *********************** Multinomial with hyperparameter tuning **************************** #
prev_score = 0
classifier = MultinomialNB(alpha = 0.1)
for alpha in np.arange(0, 1, 0.01):
    sub_classifier = MultinomialNB(alpha = alpha)
    sub_classifier.fit(X_train, y_train)
    y_pred = sub_classifier.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    if score > prev_score:
        classifier = sub_classifier
        prev_score = score
        print('Alpha : {} & Score : {}'.format(alpha, score))
 
plot_confusion_matrix(classifier, X_test, y_test, display_labels = ['Fake', 'Real'], 
                      cmap = plt.cm.Blues)

#Most Real 
sorted(zip(classifier.coef_[0], cv.get_feature_names()), reverse = True)[:20]

#Most Fake
sorted(zip(classifier.coef_[0], cv.get_feature_names()))[:20]   
    
    




