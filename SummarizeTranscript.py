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
domainKeyWords = []
lineScores = list()
keywordsToAbstractDictionary = {}


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
    global domainKeyWords
    featuresets = []
    with open('keywordsToAbstractMapping.csv') as keywordsToAbstractMappingFile:
        keywordsToAbstractMap = csv.reader(keywordsToAbstractMappingFile, delimiter=',')
        for keywordsToAbstractEntry in keywordsToAbstractMap:
            for keyword in keywordsToAbstractEntry[0].split(';'):
                featuresets += [(dialogue_act_features3(keyword), keywordsToAbstractEntry[1])]
                domainKeyWords += keyword
    train_set = featuresets[:]
    #print(train_set)
    classifier_template = nltk.NaiveBayesClassifier.train(train_set)
    pickle.dump(classifier_template, open(RHSClassifierModelName, 'wb'))

#############################################
    
def populate_domain_keywords():
    global domainKeyWords
    global keywordsToAbstractDictionary

    with open('keywordsToAbstractMapping.csv') as keywordsToAbstractMappingFile:
        keywordsToAbstractMap = csv.reader(keywordsToAbstractMappingFile, delimiter=',')
        for keywordsToAbstractEntry in keywordsToAbstractMap:
            keywordsToAbstractDictionary[keywordsToAbstractEntry[0]] = keywordsToAbstractEntry[1]
            for keyword in keywordsToAbstractEntry[0].split(';'):
                #print(keyword)
                domainKeyWords.append(keyword)
    print(keywordsToAbstractDictionary)            

#############################################

def abstruct(allTheLines):
    global lineScores

    outputFileHandler = open(str(outputFilenNo)+"_ouptput.txt", "w+")

    outputFileHandler.write("\n=====================================\n")       
    outputFileHandler.write("Abstract \n")   
    outputFileHandler.write("=======================================\n")    

    lastSpeaker = lastStatementType = lastLine = ""
    index = 0

    for eachLine in allTheLines:
        index += 1
        bFoundMatchingEntity = False
        statementlist = eachLine.split(':')
        statement = statementlist[-1].strip()

        # classify the statement as a whQuestion or ynQuestion
        questionClassificationOutput = stanfordQuestionClassifier.annotate(statement, properties={'annotators': 'parse, sentiment','outputFormat': 'json'})
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
            classified_statement = mapInputTextToLHS(statement)   

            if classifier_statement != '': 
                outputFileHandler.write(classified_statement+" : ")
                print('Input = ' + statement + ' , classified using templates as : ' + classified_statement)
                print('I found a LHS which is : ' + classified_statement)

            '''
            print(dialogue_act_features2(statement))
            classified_statement = classifier_template.classify(dialogue_act_features2(statement))
            '''
        elif classifiedType == 'Statement' or classifiedType == 'Clarify' or classifiedType == 'Other':
            if lastStatementType == 'whQuestion':
                outputFileHandler.write(statement+"\n")
                print('I found a RHS which is : ' + statement)
            else:    
                #Check for any entities to print     
                entities = nerTagger.tag(statement.split())

                for entity in entities:
                    if entity[1] != 'O':
                        bFoundMatchingEntity = True
    
                if bFoundMatchingEntity:
                    outputFileHandler.write("Notes : " + statement)
                    outputFileHandler.write("\n")
                    print('I found a note which is : ' + statement)

                for entity in entities:
                    if entity[1] != 'O':
                        #outputFileHandler.write(entity[1] + " : " + entity[0] + "\n")
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

        countOfDomainKeywords = 0    
        for domainKeyword in domainKeyWords:
            if statement.find(domainKeyword) == -1:
                continue
            else:
                countOfDomainKeywords += 1

        if bFoundMatchingEntity:
            countOfDomainKeywords += 1        

        lineScore = (statement, len(statement), questionClassificationOutput['sentences'][0]['sentimentValue'], countOfDomainKeywords, index)
        lineScores.append(lineScore)
            
        lastSpeaker = statementlist[0]
        lastStatementType = classifiedType
        lastLine = statementlist[-1]

    lineLengthArray = list(zip(*lineScores))[1]
    sentimentArray = [ int(lineScore[2]) for lineScore in lineScores ]
    keywordCountArray = list(zip(*lineScores))[3]

    #lineScore = (statement, len(statement), questionClassificationOutput['sentences'][0]['sentimentValue'], countOfDomainKeywords, 0)   
    print(lineScores)

    lineScores2 = list()
    for lineScore in lineScores:
        lineLengthNormalized = (lineScore[1] - min(lineLengthArray)) / (max(lineLengthArray) - min(lineLengthArray))
        sentimentNormalized = (int(lineScore[2]) - min(sentimentArray)) / (max(sentimentArray) - min(sentimentArray))
        keywordCountNormalized = (lineScore[3] - min(keywordCountArray)) / (max(keywordCountArray) - min(keywordCountArray))
        lineScore2 = (lineScore[0], round((0.70 * lineLengthNormalized + 0.15 * sentimentNormalized + 0.15 * keywordCountNormalized), 2), lineScore[4])
        lineScores2.append(lineScore2)

    outputFileHandler.write("\n=====================================\n")       
    outputFileHandler.write("Summary\n")   
    outputFileHandler.write("========================================\n")    

    a = sorted(lineScores2, key=lambda x: x[1], reverse=True)    

    b = sorted(a[:2], key=lambda x: x[2])    

    for lineScore2 in b:
        outputFileHandler.write(str(lineScore2[0]) + "\n")

    outputFileHandler.close()
        

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
               
def mapInputTextToLHS(inputText):
    global keywordsToAbstractDictionary
    countOfMatchingKeyWords = 0
    maxKeywordsFound = 0
    LHS = ''
    for key, value in keywordsToAbstractDictionary.items() :
        countOfMatchingKeyWords = 0
        for keyword in key.split(";"):
            if inputText.find(keyword) != -1:
                countOfMatchingKeyWords += 1
                print('found matching keyword ' + keyword + ' , countOfMatchingKeyWords = ' + str(countOfMatchingKeyWords))

        if countOfMatchingKeyWords > maxKeywordsFound:
            maxKeywordsFound = countOfMatchingKeyWords
            LHS = value        
            print('found matching keyword ' + key + ' , countOfMatchingKeyWords = ' + str(countOfMatchingKeyWords) + ' , value = ' + value)
    return LHS                 

######################################################
                
def main():
    global outputFilenNo
    global classifier_statement
    global classifier_template
    global lineScores

    fileno = "/Users/Sarnava/Data/hackathon/Hackathon_2018/transcript"

    mode = 'eval'    

    populate_domain_keywords()
    #print(domainKeyWords)
    
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
