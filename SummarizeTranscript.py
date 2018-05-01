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
import tensorflow as tf

####### Global Variable ########################
r = Rake()
nerTagger = CoreNLPNERTagger(url='http://localhost:9000')
stanfordQuestionClassifier = StanfordCoreNLP('http://localhost:9000')
stop_words = set(stopwords.words('english'))
baseDirLocation = '/Users/csdkpune/Documents/temp/Hackathon_2018'

######################################################
def getFilePath(fileName):
    return baseDirLocation+"/"+fileName

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

    with open(getFilePath("logisticRegressionModel.sav"), 'rb') as logisticRegressionModel:
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

def trainTensorflow(allInputLines):
    # number of features
    num_features = 4
    # number of target labels
    num_classes = 2
    # learning rate (alpha)
    learning_rate = 0.05
    # batch size
    batch_size = 15
    # number of epochs
    num_steps = 50

    X = tf.placeholder("float", [None, num_features], name="xInput")
    # same with labels: number of classes is known, while number of instances is left undefined
    Y = tf.placeholder("float",[None, num_classes], name="yInput")

    y_test = [[0, 1], [0, 1], [0, 1], [1, 0], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [1, 0], [0, 1], [1, 0], [1, 0], [0, 1]]

    # W - weights array
    W = tf.Variable(tf.truncated_normal([num_features, num_classes]))
    # B - bias array
    B = tf.Variable(tf.zeros([num_classes]))

    # Define a model
    # a simple linear model y=wx+b wrapped into softmax
    pY = tf.nn.softmax(tf.matmul(X, W) + B, name='outputOperation')
    # pY will contain predictions the model makes, while Y contains real data

    # Define a cost function
    cost_fn = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pY, labels=Y))
    # You could also put it in a more explicit way
    # cost_fn = -tf.reduce_sum(Y * tf.log(pY))

    # Define an optimizer
    # I prefer Adam
    opt = tf.train.AdamOptimizer(0.01).minimize(cost_fn)
    # but there is also a plain old SGD if you'd like
    #opt = tf.train.GradientDescentOptimizer(0.01).minimize(cost_fn)

    # Create and initialize a session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=1)

    num_epochs = 500
    for i in range(num_epochs):
        # run an optimization step with all train data
        sess.run(opt, feed_dict={X:getInputFeatureVector(allInputLines), Y:y_test})
        print(i)
        # thus, a symbolic variable X gets data from train_X, while Y gets data from train_Y
        # Now assess the model
        # create a variable which reflects how good your predictions are
        # here we just compare if the predicted label and the real label are the same
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pY,1), tf.argmax(Y,1)), "float"))
    # and finally, run calculations with all test data
    accuracy_value = sess.run(accuracy, feed_dict={X:getInputFeatureVector(allInputLines), Y:y_test})
    print(accuracy_value)
    saver.save(sess, getFilePath('my_test_model'), global_step=num_epochs)

######################################################

def evalTensorflow(allInputLines):
    summaryLines = list()
    with tf.Session() as sess1:    
        saver = tf.train.import_meta_graph(getFilePath('my_test_model-500.meta'))
        saver.restore(sess1, tf.train.latest_checkpoint(baseDirLocation))
        graph = tf.get_default_graph()
        a = tf.get_default_graph().get_tensor_by_name('outputOperation:0')
        x_input = tf.get_default_graph().get_tensor_by_name('xInput:0')
        v = sess1.run(a, feed_dict={x_input:getInputFeatureVector(allInputLines)})
        print(v)
        ouputTensorflow = tf.argmax(v, 1).eval()

    index = 0
    for inputLine in allInputLines:
        if (ouputTensorflow[index] == 0):
            statementlist = inputLine.split(':')
            statement = statementlist[-1].strip()
            print(statement)
            summaryLines.append(statement)
        index+=1
    return summaryLines    

######################################################

def main():
    global outputFilenNo

    fileno = getFilePath("transcript")
    summaryLines = list()
    with  open(str(fileno)+".txt", "r") as inputFileHandler:
        lines = inputFileHandler.readlines()
        #trainTensorflow(lines)
        summaryLines = evalTensorflow(lines)
        print(summaryLines)
        #summaryLines = eval(lines)
        #print(summaryLines)
        printOutputFile(summaryLines)
        
    with open(str(fileno)+"_ouptput.txt", "w+") as outputFileHandler:
        for summaryLine in summaryLines:
            outputFileHandler.write(summaryLine+"\n")   
            

##############################################################        
if __name__ == "__main__": main()
