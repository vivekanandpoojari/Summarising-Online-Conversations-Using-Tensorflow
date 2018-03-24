from __future__ import division
import inspect
import nltk
from os import listdir
from nltk import word_tokenize, pos_tag
from nltk.corpus import nps_chat
from time import time,sleep
from collections import Counter

alo =6
d = {}
posans= []
negans = []

d["Is the Lan connected ?"]= 0
posans.append("The Lan Is connected.")
negans.append("The Lan is not connected.")

ans = ""
posts = nps_chat.xml_posts()[:10000]

def abstruct(value):
	global ans, d, posant, negans
	flag = userasked= 0
	tempvalue=""
	file = open(str(alo)+"_ouptput.txt", "w+")
	for f in line:
		#print("flag: ", flag)
		a = f.split(':')
		temp = a[-1].strip()
		dac = classifier.classify(dialogue_act_features(temp))
		#print (a,dac)
		
		if userasked ==1:
			ans += temp
			file.write(temp+"\n");
			continue
	
		if dac == "Greet" or dac == "System":
			continue
		elif dac == "Statement" or dac == "Emphasis":
			ans += temp
			file.write(temp+"\n"); 
			
		elif dac == "ynQuestion" or dac == "whQuestion":
			if a[0] == 'U':
				ans += temp
				file.write(temp+"\n");
				userasked=1
				continue
			elif a[0] == 'A':
				flag = 1
				tempvalue = temp
                		continue
        	elif dac == "yAnswer":
			if flag == 1:
				#print(d[tempvalue])
				ans += posans[d[tempvalue]]
				file.write(posans[d[tempvalue]]+"\n");
			else: 
				ans += temp
				file.write(temp+"\n");
		flag = userasked =0
			
	file.close()
	
	return 

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
'''
print('Welecome to TATA Docomo ' + classifier.classify(dialogue_act_features('Welcome to TATA Docomo')))
print('Is your lan cable connected ? ' + classifier.classify(dialogue_act_features('Is your lan cable connected ?')))
print('Yes ,' + classifier.classify(dialogue_act_features('Yes')))
print('What is the ETA for this fix ? ' + classifier.classify(dialogue_act_features('What is the ETA for this fix ?')))
print('It is expected to be fixed by Friday ' + classifier.classify(dialogue_act_features('It is expected to be fixed by Friday')))
print('I am happy with this service ' + classifier.classify(dialogue_act_features('I am happy with this service')))
print('This service is very poor ' + classifier.classify(dialogue_act_features('This service is very poort')))
print('No i ehhh ' + classifier.classify(dialogue_act_features('No')))
print('I am Sarnava?, ' + classifier.classify(dialogue_act_features('okay what is the number? ')))

'''
foo= open(str(alo)+".txt", "r")
line = foo.readlines()

abstruct(line)
foo.close()
print(ans)
