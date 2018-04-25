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
from pycorenlp import StanfordCoreNLP
from nltk.tree import Tree


####### Global Variable ########################
outputFilenNo = 0
typeofdialogue =[]
classifier_statement = classifier_template = 0
posts = nps_chat.xml_posts()[:]
r = Rake()
keyWordsPerInputSentence =[]
nerTagger = CoreNLPNERTagger(url='http://localhost:9000')
stanfordQuestionClassifier = StanfordCoreNLP('http://localhost:9000')

inputTrainingCorpus = {'LAN Connection Status':['is your lan cable connected'], 
                         'Modem working status':['is your modem modem light blinking?'],
                         'Modem working status':['Is your modem on?'],
                         'Modem restarted ':['Did you restart the modem ?'],
                         'Wifi working status':['wifi is not working'],
                         'LAN connection status':['Is the Lan Connected'],
                         'Internet working status':['can you browse google?'],
                         'Internet working status':['Is the internet light on'],
                         'Ticket ID':['What is the ticket id'],
                         'Estimated Date for fix':['the eta for this problem is'],
                         'Estimated Date for fix':['what is the estimated time'],
                         'Order Location':['where is my order'],
                         'Service required':['How may I help you ?'],
                         'Service required':['Is there anything I can help you with ?'],
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
                         #'e-bill Status':['Do you want an ebill'],
                         #'e-bill Status':['your ebill has been send']
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
    LHS = False
    for eachLine in allTheLines:
        
        statementlist = eachLine.split(':')
        statement = statementlist[-1].strip()

        questionClassificationOutput = stanfordQuestionClassifier.annotate(statement, properties={'annotators': 'parse','outputFormat': 'json'})
        questionClassificationTree = Tree.fromstring(questionClassificationOutput['sentences'][0]['parse'])
        questionClassificationTreeProductions = questionClassificationTree.productions()
        standfordQuestionClassification = ''

        for questionClassificationTreeLevel in questionClassificationTreeProductions:
            if (str(questionClassificationTreeLevel.lhs()) == 'ROOT' and str(questionClassificationTreeLevel.rhs()[0]) == 'SBARQ'):
                standfordQuestionClassification = 'whQuestion'
            elif (str(questionClassificationTreeLevel.lhs()) == 'ROOT' and str(questionClassificationTreeLevel.rhs()[0]) == 'SQ'):
                standfordQuestionClassification = 'ynQuestion'
        
        classifiedType = classifier_statement.classify(dialogue_act_features(statement))
        
        # Overwrite question classification using stanford since its more reliable than nps chat corpus
        if standfordQuestionClassification == 'whQuestion' or standfordQuestionClassification == 'ynQuestion':
            classifiedType = standfordQuestionClassification
        
        #print(statement + " ||  " + classifiedType)
        # + " : FoundLHS ? = " + str(LHS))
        
        if classifiedType not in typeofdialogue:
            typeofdialogue.append(classifiedType)
        
        if classifiedType == "":
            continue
        elif classifiedType == "Greet" or classifiedType == "System":
            continue
        elif classifiedType == "ynQuestion" or classifiedType == 'whQuestion':
            classified_statement = classifier_template.classify(dialogue_act_features(statement))
            classified_statementProb = classifier_template.prob_classify(dialogue_act_features(statement))
            #print(classified_statementProb.prob(list(classified_statementProb.samples())[0]))
            #print(len(classified_statementProb.samples()))
            #if (classified_statementProb.prob(list(classified_statementProb.samples())[0]) * len(classified_statementProb.samples())) == 1:
            outputFileHandler.write(classified_statement+" : ")
            print('Input = ' + statement + ' , classified using templates as : ' + classified_statement)
            print('I found a LHS which is : ' + classified_statement)
            LHS = True
        elif classifiedType == 'Statement' or classifiedType == 'Clarify':
             if lastStatementType == 'whQuestion':
                  #NER and paste
                  entities = nerTagger.tag(statement.split())
                  bFoundMatchingEntity = False
                  for entity in entities:
                       if entity[1] == 'CITY' or entity[1] == 'DURATION' or entity[1] == 'NUMBER':
                            outputFileHandler.write(entity[0] + " ")
                            print('I found a RHS which is : ' + entity[0])
                            bFoundMatchingEntity = True
                  if not bFoundMatchingEntity:
                       outputFileHandler.write(statement)
                       print('I found a RHS which is : ' + statement)
                  outputFileHandler.write("\n")
                  LHS = False
             else:
                  if not LHS:
                       classified_statement = classifier_template.classify(dialogue_act_features(statement))
                       #outputFileHandler.write(classified_statement+"\n")
                       classified_statementProb = classifier_template.prob_classify(dialogue_act_features(statement))
                       if (classified_statementProb.prob(list(classified_statementProb.samples())[0]) * len(classified_statementProb.samples())) == 1:
                           outputFileHandler.write(statement+"\n")
                           print('I found a RHS which is : ' + statement)
                           LHS = False
                       else:
                           #outputFileHandler.write(classified_statement+" : ")
                           outputFileHandler.write("Notes : " + statement)
                           outputFileHandler.write("\n")
                           print('I found a note which is : ' + statement)
                           LHS = False
                  else:
                       outputFileHandler.write(statement+"\n")
                       print('I found a RHS which is : ' + statement)
                       LHS = False
        elif classifiedType == 'yAnswer' and lastStatementType == 'ynQuestion':
            print('Found yes answer to ynQuestion')
            outputFileHandler.write('Yes'+"\n")
            print('I found a RHS which is : ' + statement)
            LHS = False
        elif classifiedType == 'nAnswer' and lastStatementType == 'ynQuestion':
            print('Found no answer to ynQuestion')
            outputFileHandler.write('No'+"\n")
            print('I found a RHS which is : ' + statement)
            LHS = False
        else:
            outputFileHandler.write(statement+"\n")
            print('I found a RHS which is : ' + statement)
            LHS = False
            
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
    
    filesToProcess = [1]

    for fileno in filesToProcess:
        outputFilenNo = fileno
        ans = ""
        #print(fileno)
        
        inputFileHandler = open(str(fileno)+".txt", "r")
        
        lines = inputFileHandler.readlines()
        abstruct(lines)
        inputFileHandler.close()
        #printOutputFile()
        
##############################################################        
if __name__ == "__main__": main()
