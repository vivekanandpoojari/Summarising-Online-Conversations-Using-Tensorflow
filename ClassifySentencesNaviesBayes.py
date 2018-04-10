
# coding: utf-8

# In[96]:


from rake_nltk import Rake
import nltk
import pickle as pk
r = Rake()


# In[97]:


inputTrainingCorpus = {'LAN Connection Status':['is your lan cable connected'], 
                         'Modem working status':['is your modem modem light blinking?'],
                         'Modem working status':['Is your modem on?'],
                         'Wifi working status':['wifi is not working'],
                         'LAN connection status':['Is the Lan Connected'],
                         'Internet working status':['can you browse google?'],
                         'Internet working status':['Is the internet light on'],
                         'Ticket ID': ['your ticket id is'],
                         'Estimated Date for fix':['the eta for this problem is'],
                         'Order Location':['where is my order'],
                         'Service required':['How may I help you'],
                         'Service required':['Is there anything I can help you with'],
                         'Modem working status':['Is your modem on?'],
                         'Mobile Data status':['Mobile data turned of'],
                         'Recharge status':['When did you last recharge'],
                         'Recharge status':['Did you recharge'],
                         'Recharge status':['What is my last recharge'],
                         'Phone number':['what is the registered number'],
                         'Connection Status':['What is the connection type']
                       }
            

# In[100]:


def dialogue_act_features(inputTrainingSentence):
    features = {}
    for word in nltk.word_tokenize(inputTrainingSentence):
        features['contains({})'.format(word.lower())] = True
    return features


# In[108]:


features = {}
featuresets = []
keyWordsPerInputSentence = list()

for abstractCategory, inputTrainingSentences in inputTrainingCorpus.items():
    for inputTrainingSentence in inputTrainingSentences:
        r.extract_keywords_from_text(inputTrainingSentence)
        inputTrainingSentenceKeyWords = ' '.join(r.get_ranked_phrases())
        keyWordsPerInputSentence.append(inputTrainingSentenceKeyWords)
        featuresets += [(dialogue_act_features(inputTrainingSentenceKeyWords), abstractCategory)]

    


# In[109]:


#print(featuresets)


# In[113]:


size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)
testsentence = 'Do you have prepaid or postpaid'
print(classifier.classify(dialogue_act_features(testsentence)))

print("Probability :------------------")
dist = classifier.prob_classify(dialogue_act_features(testsentence))
#print(list(dist.samples())[0])
#print(dist.prob(list(dist.samples())[0]) * len(dist.samples()))
for label in dist.samples():
    print("%s: %f" % (label, dist.prob(label)))



'''
testsentence = 'The eta of this fix is'
print(classifier.classify(dialogue_act_features(testsentence)))

print("Probability :------------------")
dist = classifier.prob_classify(dialogue_act_features(testsentence))
#print(dist.prob(dist.samples()[0]) * len(dist.samples()))
for label in dist.samples():
    print("%s: %f" % (label, dist.prob(label)))
'''
