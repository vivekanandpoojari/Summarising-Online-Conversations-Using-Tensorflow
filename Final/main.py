############# Imports #########################
from __future__ import division
import inspect
import nltk
from os import listdir
from nltk import word_tokenize, pos_tag
from nltk.corpus import nps_chat
from time import time,sleep
from rake_nltk import Rake
from collections import Counter


####### Global Variable ########################
outputFilenNo = 0
typeofdialogue =[]
classifier_statement = classifier_template = 0
posts = nps_chat.xml_posts()[:]
r = Rake()
keyWordsPerInputSentence =[]

inputTrainingCorpus = {'LAN Connection Status':['is your lan cable connected'], 
                         'Modem working status':['is your wifi modem light blinking?'], 
                         'Internet working status':['can you browse google?'],
                         'Ticket ID': ['your ticket id is'],
                         'Estimated Date for fix':['the eta for this problem is'],
                         'Order Location':['where is my order'],
                         'Service required':['How can I help you']
                         }

######################################################

def dialogue_act_features(post):
     features = {}
     for word in word_tokenize(post):
         features['contains({})'.format(word.lower())] = True
     return features


######################################################

def train_npschat():
    global classifier_statement
    featuresets = [(dialogue_act_features(post.text), post.get('class'))for post in posts] 
    size = int(len(featuresets) * 0.1)
    train_set, test_set = featuresets[size:], featuresets[:size]
    classifier_statement = nltk.NaiveBayesClassifier.train(train_set)

#############################################
    
def train_template():
    global classifier_template, keyWordsPerInputSentence
    featuresets = []
    for abstractCategory, inputTrainingSentences in inputTrainingCorpus.items():
        for inputTrainingSentence in inputTrainingSentences:
            r.extract_keywords_from_text(inputTrainingSentence)
            inputTrainingSentenceKeyWords = ' '.join(r.get_ranked_phrases())
            keyWordsPerInputSentence.append(inputTrainingSentenceKeyWords)
            featuresets += [(dialogue_act_features(inputTrainingSentenceKeyWords), abstractCategory)]
    size = int(len(featuresets) * 0.1)
    train_set, test_set = featuresets[size:], featuresets[:size]
    classifier_template = nltk.NaiveBayesClassifier.train(train_set)


#############################################
def abstruct(allTheLines):
    
    outputFileHandler = open(str(outputFilenNo)+"_ouptput.txt", "w+")
    
    for eachLine in allTheLines:
        
        statement = eachLine.split(':')
        statement = statement[-1].strip()
        
        classifiedType = classifier_statement.classify(dialogue_act_features(statement))
        print(statement)
        print(classifiedType)
        if classifiedType not in typeofdialogue:
            typeofdialogue.append(classifiedType)
        
        if classifiedType == "":
            continue
        elif classifiedType == "Greet" or classifiedType == "System":
            continue
        elif classifiedType == "ynQuestion" or classifiedType == 'whQuestion':
            classified_statement = classifier_template.classify(dialogue_act_features(statement))
            outputFileHandler.write(classified_statement+" : ")
        elif classifiedType == 'Statement':
            classified_statement = classifier_template.classify(dialogue_act_features(statement))
            outputFileHandler.write(classified_statement+"\n")
        elif classifiedType == 'yAnswer':
            outputFileHandler.write('Yes'+"\n")
        elif classifiedType == 'nAnswer':
            outputFileHandler.write('No'+"\n")
        else:
            outputFileHandler.write(statement+"\n")
            
        

######################################################
    
            
def main():
    global outputFilenNo
    
    train_npschat()
    train_template()
    
    for fileno in range(1,2):
        
        outputFilenNo = fileno
        ans = ""
        print(fileno)
        
        inputFileHandler = open(str(fileno)+".txt", "r")
        
        lines = inputFileHandler.readlines()
        abstruct(lines)
        inputFileHandler.close()
        
        print(ans)
        
##############################################################        
if __name__ == "__main__": main()