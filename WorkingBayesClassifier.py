from __future__ import division
import inspect
import nltk
from os import listdir
from nltk import word_tokenize, pos_tag
from nltk.corpus import nps_chat
from time import time,sleep
from collections import Counter

posts = nps_chat.xml_posts()[:10000]

def dialogue_act_features(post):
     features = {}
     for word in word_tokenize(post):
         features['contains({})'.format(word.lower())] = True
     return features

#for post in posts:
#	print(dialogue_act_features(post.text))
	#print(" Post Text : " + post.text)
	#print(" Post class = " + post.get('class'))

featuresets = [(dialogue_act_features(post.text), post.get('class'))
                for post in posts] 

size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
#print(train_set)
#print("-----------")
#print(test_set)
classifier = nltk.NaiveBayesClassifier.train(train_set)
print('Welecome to TATA Docomo ' + classifier.classify(dialogue_act_features('Welcome to TATA Docomo')))
print('Is your lan cable connected ? ' + classifier.classify(dialogue_act_features('Is your lan cable connected ?')))
print('Yes ' + classifier.classify(dialogue_act_features('Yes')))
print('What is the ETA for this fix ? ' + classifier.classify(dialogue_act_features('What is the ETA for this fix ?')))
print('It is expected to be fixed by Friday ' + classifier.classify(dialogue_act_features('It is expected to be fixed by Friday')))
print('I am happy with this service ' + classifier.classify(dialogue_act_features('I am happy with this service')))
print('This service is very poor ' + classifier.classify(dialogue_act_features('This service is very poort')))
print('No i ehhh ' + classifier.classify(dialogue_act_features('No')))
print('ohhh  zzzz hmmm ' + classifier.classify(dialogue_act_features('ohhh  zzzz hmmm ')))

# A :      