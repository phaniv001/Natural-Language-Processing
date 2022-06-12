# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 12:51:56 2022

@author: laksh
"""
import nltk, re
from nltk.corpus import stopwords
from nltk import word_tokenize, sent_tokenize, pos_tag

# Pre-Process the data"

corpus = []
#Quest = """What is your mother’s name?"""
#Quest = "What is the salary of Lakshman Vuyyuri"
#Quest = "What is the work location of Lakshman Vuyyuri"
Quest = "How many employees are working from India, US, China, Mexico"
review = re.sub('[^a-zA-Z0-9]', ' ', Quest)
review = review.lower()
review = review.split()
review = [word for word in review if word not in set(stopwords.words('english'))]
review = ' '.join(review)
#corpus.append(review)

#Name Entity Recognition
tokens = word_tokenize(Quest)
tag = pos_tag(tokens)
print(tag)

ne_tree = nltk.ne_chunk(tag)
print(ne_tree)






