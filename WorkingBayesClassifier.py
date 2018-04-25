from __future__ import division
import inspect
import nltk
from os import listdir
from nltk import word_tokenize, pos_tag
from nltk.corpus import nps_chat
from time import time,sleep
from collections import Counter
typeofdialogue =[]
alo =0
d = {}
posans= []
negans = []

removeword = ["Thank you", "Thanks", "Welcome","Please"]

C_repls = ('I ', 'Customer '), ('my ', 'Customers '), ('me ','Customers '), ('mine ','Customers ')
A_repls = ('I ', 'Agent '), ('my ', 'Agent '), ('me ','Agent '), ('mine ','Agent ')
d["Is the Lan connected ?"]= 0
posans.append("The Lan Is connected.")
negans.append("The Lan is not connected.")

ans = ""

l=[]
l= nps_chat.xml_posts()[:]
posts = nps_chat.xml_posts()[:]


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

inputSenetences = ['Welcome to TATA Docomo',
					'Is your lan cable connected ',
					'Is your lan cable connected ?',
					'How may I help you']

for text in inputSenetences:
	print("text = " + text + " , classicfication = " + classifier.classify(dialogue_act_features(text)))
