
# coding: utf-8

# In[96]:


from rake_nltk import Rake
import nltk
import pickle as pk
r = Rake()


# In[97]:


inputTrainingCorpus = {'LAN Connection Status':['is your lan cable connected'], 
                         'Modem working status':['is your wifi modem light blinking?'], 
                         'Internet working status':['can you browse google?'],
                         'Ticket ID': ['you ticket id is'],
                         'Estimated Date for fix':['the eta for this problem is']
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


print(featuresets)


# In[113]:


size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)
testsentence = 'is your modem light blinking '
classifier.classify(dialogue_act_features(testsentence))

