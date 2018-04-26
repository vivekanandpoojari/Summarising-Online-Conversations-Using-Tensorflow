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
import csv
import pickle

####### Global Variable ########################
outputFilenNo = 0
typeofdialogue =[]
classifier_statement = classifier_template = 0
RHSClassifierModelName = '/Users/Sarnava/Data/hackathon/Hackathon_2018/rhsClassifierModel.sav'
NPSDialogActClassifierModelName = '/Users/Sarnava/Data/hackathon/Hackathon_2018/NPSDialogActClassifierModel.sav'
posts = nps_chat.xml_posts()[:]
r = Rake()
nerTagger = CoreNLPNERTagger(url='http://localhost:9000')
stanfordQuestionClassifier = StanfordCoreNLP('http://localhost:9000')


######################################################

def dialogue_act_features(post):
     features = {}
     for word in word_tokenize(post):
         features['contains({})'.format(word.lower())] = True
     return features

######################################################

def dialogue_act_features2(inputText):
    features = {}
    r.extract_keywords_from_text(inputText)
    inputTrainingSentenceKeyWords = ' '.join(r.get_ranked_phrases())

    for word in word_tokenize(inputTrainingSentenceKeyWords):
         features['{}'.format(word.lower())] = True
    return features

######################################################

def dialogue_act_features3(inputText):
    features = {}
    features['{}'.format(inputText.lower())] = True
    return features         

######################################################

def train_npschat():
    global classifier_statement
    featuresets = []
    for post in posts:
        featuresets += [(dialogue_act_features2(post.text), post.get('class'))]
    train_set = featuresets[:]
    classifier_statement = nltk.NaiveBayesClassifier.train(train_set)
    pickle.dump(classifier_statement, open(NPSDialogActClassifierModelName, 'wb'))

#############################################
    
def train_template():
    global classifier_template
    featuresets = []
    with open('keywordsToAbstractMapping.csv') as keywordsToAbstractMappingFile:
        keywordsToAbstractMap = csv.reader(keywordsToAbstractMappingFile, delimiter=',')
        for keywordsToAbstractEntry in keywordsToAbstractMap:
            for keyword in keywordsToAbstractEntry[0].split(';'):
                featuresets += [(dialogue_act_features3(keyword), keywordsToAbstractEntry[1])]
    train_set = featuresets[:]
    print(train_set)
    classifier_template = nltk.NaiveBayesClassifier.train(train_set)
    pickle.dump(classifier_template, open(RHSClassifierModelName, 'wb'))

#############################################

def abstruct(allTheLines):
    outputFileHandler = open(str(outputFilenNo)+"_ouptput.txt", "w+")
    lastSpeaker = lastStatementType = lastLine = ""

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
        
        classifiedType = classifier_statement.classify(dialogue_act_features2(statement))
        
        # Overwrite question classification using stanford since its more reliable than nps chat corpus
        if standfordQuestionClassification == 'whQuestion' or standfordQuestionClassification == 'ynQuestion':
            classifiedType = standfordQuestionClassification
        
        print(statement + " ||  " + classifiedType)
        
        if classifiedType not in typeofdialogue:
            typeofdialogue.append(classifiedType)
        
        if classifiedType == "":
            continue
        elif classifiedType == "Greet" or classifiedType == "System":
            continue
        elif classifiedType == "ynQuestion" or classifiedType == 'whQuestion':
            classified_statement = classifier_template.classify(dialogue_act_features2(statement))
            classified_statementProb = classifier_template.prob_classify(dialogue_act_features2(statement))
            outputFileHandler.write(classified_statement+" : ")
            print('Input = ' + statement + ' , classified using templates as : ' + classified_statement)
            print('I found a LHS which is : ' + classified_statement)
        elif classifiedType == 'Statement' or classifiedType == 'Clarify' or classifiedType == 'Other':
            if lastStatementType == 'whQuestion':
                outputFileHandler.write(statement+"\n")
                print('I found a RHS which is : ' + statement)
                continue

            #Check for any entities to print     
            entities = nerTagger.tag(statement.split())
            bFoundMatchingEntity = False
            for entity in entities:
                if entity[1] == 'CITY' or entity[1] == 'DURATION' or entity[1] == 'NUMBER':
                    bFoundMatchingEntity = True
    
            if bFoundMatchingEntity:
                outputFileHandler.write("Notes : " + statement)
                outputFileHandler.write("\n")
                print('I found a note which is : ' + statement)

                for entity in entities:
                    if entity[1] == 'CITY' or entity[1] == 'DURATION' or entity[1] == 'NUMBER':
                        outputFileHandler.write(entity[1] + " : " + entity[0] + "\n")
                        print('I found a RHS which is : ' + entity[0])
                        bFoundMatchingEntity = True
            
        elif classifiedType == 'yAnswer' and lastStatementType == 'ynQuestion':
            print('Found yes answer to ynQuestion')
            outputFileHandler.write('Yes'+"\n")
            print('I found a RHS which is : ' + statement)
        elif classifiedType == 'nAnswer' and lastStatementType == 'ynQuestion':
            print('Found no answer to ynQuestion')
            outputFileHandler.write('No'+"\n")
            print('I found a RHS which is : ' + statement)
        else:
            outputFileHandler.write(statement+"\n")
            print('I found a RHS which is : ' + statement)
            
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
    global classifier_statement
    global classifier_template

    fileno = "/Users/Sarnava/Data/hackathon/Hackathon_2018/transcript"

    mode = 'eval'    
    
    if mode == 'train':
        train_npschat()
        train_template()
    elif mode == 'eval':
        with open(RHSClassifierModelName, 'rb') as f:
            classifier_template = pickle.load(f)
        with open(NPSDialogActClassifierModelName, 'rb') as g:
            classifier_statement = pickle.load(g)
    
    filesToProcess = [1]
    outputFilenNo = fileno
    ans = ""
    inputFileHandler = open(str(fileno)+".txt", "r")
    lines = inputFileHandler.readlines()
    abstruct(lines)
    inputFileHandler.close()
    printOutputFile()
    
        
##############################################################        
if __name__ == "__main__": main()
