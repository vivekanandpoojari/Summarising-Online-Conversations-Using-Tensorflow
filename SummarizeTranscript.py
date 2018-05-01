############# Imports #########################
from __future__ import division
import inspect
import nltk
from os import listdir
from nltk import word_tokenize, pos_tag
from rake_nltk import Rake
from collections import Counter
from nltk.tag.stanford import CoreNLPNERTagger
from tkinter import *
from pycorenlp import StanfordCoreNLP
from nltk.tree import Tree
import pickle
from nltk.corpus import stopwords
import numpy as np
from sklearn import linear_model

####### Global Variable ########################
r = Rake()
nerTagger = CoreNLPNERTagger(url='http://localhost:9000')
stanfordQuestionClassifier = StanfordCoreNLP('http://localhost:9000')
stop_words = set(stopwords.words('english'))

######################################################

def getCountOfKeywords(inputText):
    r.extract_keywords_from_text(inputText)
    inputTrainingSentenceKeyWords = ' '.join(r.get_ranked_phrases())
    countOfKeywords = 0

    for word in word_tokenize(inputTrainingSentenceKeyWords):
        if word not in stop_words:
         countOfKeywords += 1
    return countOfKeywords

######################################################

def getInputFeatureVector(inputLines):
    lineScores = list()
    lastStatementType = ""
    lastStatement = ""
    isAnswerToServiceRequired = False
    index = 0

    for eachLine in inputLines:
        isAnswerToServiceRequired = False
        bFoundMatchingEntity = False
        statementlist = eachLine.split(':')
        statement = statementlist[-1].strip()

        index+=1

        if (lastStatementType == 'whQuestion') and lastStatement.find('help') != -1 :
            isAnswerToServiceRequired = True

        # classify the statement as a whQuestion or ynQuestion
        questionClassificationOutput = stanfordQuestionClassifier.annotate(statement, properties={'annotators': 'parse, sentiment','outputFormat': 'json'})
        questionClassificationTree = Tree.fromstring(questionClassificationOutput['sentences'][0]['parse'])
        questionClassificationTreeProductions = questionClassificationTree.productions()
        
        classifiedType = ''    

        for questionClassificationTreeLevel in questionClassificationTreeProductions:
            if (str(questionClassificationTreeLevel.lhs()) == 'ROOT' and str(questionClassificationTreeLevel.rhs()[0]) == 'SBARQ'):
                classifiedType = 'whQuestion'
                break
        
        countOfDomainKeywords = getCountOfKeywords(statement)    

        #Check for any entities to print     
        entities = nerTagger.tag(statement.split())

        for entity in entities:
            if entity[1] != 'O':
                countOfDomainKeywords += 1            

        lineScore = (len(statement), questionClassificationOutput['sentences'][0]['sentimentValue'], countOfDomainKeywords, int(isAnswerToServiceRequired))
        lineScores.append(lineScore)
        lastStatementType = classifiedType
        lastStatement = statement

    lineLengthArray = list(zip(*lineScores))[0]
    sentimentArray = [ int(lineScore[1]) for lineScore in lineScores ]
    keywordCountArray = list(zip(*lineScores))[2]

    x_test = list()
    
    for lineScore in lineScores:
        lineLengthNormalized = round((lineScore[0] - min(lineLengthArray)) / (max(lineLengthArray) - min(lineLengthArray)), 3)

        if (max(sentimentArray) - min(sentimentArray)) == 0:
            sentimentNormalized = 0
        else:          
            sentimentNormalized = round((int(lineScore[1]) - min(sentimentArray)) / (max(sentimentArray) - min(sentimentArray)), 3)

        if (max(keywordCountArray) -  min(keywordCountArray) == 0):
            keywordCountNormalized = 0
        else:
            keywordCountNormalized = round((lineScore[2] - min(keywordCountArray)) / (max(keywordCountArray) - min(keywordCountArray)), 3)

        x_test.append([lineLengthNormalized, sentimentNormalized, keywordCountNormalized, lineScore[3]])

    return x_test    

######################################################

def train(allTheLines):    
    x_test = getInputFeatureVector(allTheLines)
    y_test = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]

    logreg = linear_model.LogisticRegression(class_weight='balanced')
    logreg.fit(x_test, y_test)
    pickle.dump(logreg, open("/Users/Sarnava/Data/hackathon/Hackathon_2018/logisticRegressionModel.sav", 'wb'))
        
######################################################
                
def eval(allTheLines):    
    x_test = getInputFeatureVector(allTheLines)
    y = ''

    with open("/Users/Sarnava/Data/hackathon/Hackathon_2018/logisticRegressionModel.sav", 'rb') as logisticRegressionModel:
        model = pickle.load(logisticRegressionModel)
        y = model.predict(x_test)

    index = 0
    summaryLines = list()
    for inputLine in allTheLines:
        if (y[index] == 1):
            statementlist = inputLine.split(':')
            statement = statementlist[-1].strip()
            summaryLines.append(statement)
        index+=1
    return summaryLines    
        
######################################################

def printOutputFile(lines):     
    root = Tk()
     
    #printFileHandler = open(str(outputFileName)+"_ouptput.txt", "r")
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

    fileno = "/Users/Sarnava/Data/hackathon/Hackathon_2018/transcript"
    summaryLines = list()
    with  open(str(fileno)+".txt", "r") as inputFileHandler:
        lines = inputFileHandler.readlines()
        #train(lines)
        summaryLines = eval(lines)
        printOutputFile(summaryLines)

    with open(str(fileno)+"_ouptput.txt", "w+") as outputFileHandler:
        for summaryLine in summaryLines:
            outputFileHandler.write(summaryLine+"\n")   

##############################################################        
if __name__ == "__main__": main()
