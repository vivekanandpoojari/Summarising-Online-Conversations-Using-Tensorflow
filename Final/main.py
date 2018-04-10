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
from nltk.tag.stanford import CoreNLPNERTagger
from tkinter import *


####### Global Variable ########################
outputFilenNo = 0
typeofdialogue =[]
classifier_statement = classifier_template = 0
posts = nps_chat.xml_posts()[:]
r = Rake()
keyWordsPerInputSentence =[]
nerTagger = CoreNLPNERTagger(url='http://localhost:9000')

inputTrainingCorpus = {'LAN Connection Status':['is your lan cable connected'], 
                         'Modem working status':['is your modem modem light blinking?'],
                         'Modem working status':['Is your modem on?'],
                         'Wifi working status':['wifi is not working'],
                         'LAN connection status':['Is the Lan Connected'],
                         'Internet working status':['can you browse google?'],
                         'Internet working status':['Is the internet light on'],
                         'Ticket ID':['Do you want to raise a ticket'],
                         'Ticket ID':['What is the ticket id'],
                         'Estimated Date for fix':['the eta for this problem is'],
                         'Estimated Date for fix':['what is the estimated time'],
                         'Order Location':['where is my order'],
                         'Service required':['How may I help you'],
                         'Service required':['Is there anything I can help you with'],
                         'Modem working status':['Is your modem on?'],
                         'Mobile Data status':['Mobile data turned of'],
                         'Recharge status':['When did you last recharge'],
                         'Recharge status':['Did you recharge'],
                         'Recharge status':['What is my last recharge'],
                         'Phone number':['what is the registered number'],
                         'Connection Status':['What is the connection type'],
                         'Email Id':['What is your registered email id'],
                         'Email Id':['What is your mail id' ],
                         'Service Termination':['I want to stop the service'],
                         'Service Termination':['I want to discontinue'],
                         'Service Termination':['stop the service'],
                         'e-bill Status':['Do you want an ebill'],
                         'e-bill Status':['your ebill has been send']
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
    size = int(len(featuresets) * 0.9)
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
    lastSpeaker = lastStatementType = lastLine = ""
    
    for eachLine in allTheLines:
        
        statementlist = eachLine.split(':')
        statement = statementlist[-1].strip()
        
        classifiedType = classifier_statement.classify(dialogue_act_features(statement))
        
        
        print(statement)
        print(classifiedType)
        if classifiedType not in typeofdialogue:
            typeofdialogue.append(classifiedType)
        
        if classifiedType == "":
            continue
        elif classifiedType == "Greet" or classifiedType == "System":
            continue
        elif classifiedType == "ynQuestion" or classifiedType == 'whQuestion' or classifiedType == 'Clarify':
            classified_statement = classifier_template.classify(dialogue_act_features(statement))
            outputFileHandler.write(classified_statement+" : ")          
        elif classifiedType == 'Statement':
             if lastStatementType == 'whQuestion' or lastStatementType == 'Clarify':
                  #NER and paste
                  entities = nerTagger.tag(statement.split())
                  bFoundMatchingEntity = False
                  for entity in entities:
                       if entity[1] == 'CITY' or entity[1] == 'DURATION' or entity[1] == 'NUMBER':
                            outputFileHandler.write(entity[0] + " ")
                            bFoundMatchingEntity = True
                  if not bFoundMatchingEntity:
                       outputFileHandler.write(statement)
                  outputFileHandler.write("\n")                      
             else:
                  classified_statement = classifier_template.classify(dialogue_act_features(statement))
                  #outputFileHandler.write(classified_statement+"\n")
                  classified_statementProb = classifier_template.prob_classify(dialogue_act_features(statement))
                  if (classified_statementProb.prob(list(classified_statementProb.samples())[0]) * len(classified_statementProb.samples())) == 1:
                      outputFileHandler.write(statement+"\n")
                  else:
                      outputFileHandler.write(classified_statement+"\n")
        elif classifiedType == 'yAnswer' and lastStatementType == 'ynQuestion':
            outputFileHandler.write('Yes'+"\n")
        elif classifiedType == 'nAnswer' and lastStatementType == 'ynQuestion':
            outputFileHandler.write('No'+"\n")
        else:
            outputFileHandler.write(statement+"\n")
            
        lastSpeaker = statementlist[0]
        lastStatementType = classifiedType
        lastLine = statementlist[-1]
        

######################################################

def printOutputFile():
     
     root = Tk()
     
     printFileHandler = open(str(outputFilenNo)+"_ouptput.txt", "r")
     
     lines = printFileHandler.readlines()
     heightOfBox = len(lines) + 20
     T = Text(root, height=heightOfBox, width=100)
     T.pack()
     
     for line in lines:
          T.insert(END, line)
          T.insert(END, "\n")
     b2 = Button(root, text='Quit', command=root.quit)
     b2.pack(side=RIGHT, padx=15, pady=15)
     mainloop()
     
     

######################################################
                
def main():
    global outputFilenNo
    
    train_npschat()
    train_template()
    
    for fileno in range(4,5):
        
        outputFilenNo = fileno
        ans = ""
        print(fileno)
        
        inputFileHandler = open(str(fileno)+".txt", "r")
        
        lines = inputFileHandler.readlines()
        abstruct(lines)
        inputFileHandler.close()
        printOutputFile()
        
##############################################################        
if __name__ == "__main__": main()
