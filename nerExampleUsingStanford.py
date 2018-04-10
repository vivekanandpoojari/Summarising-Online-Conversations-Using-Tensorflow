# -*- coding: utf-8 -*-

from nltk.tag.stanford import CoreNLPNERTagger

'''
Start the stanford server uisng the command java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer
This command should be called from the directory where the stanford nlp is downloaded
The default port picked by server is 9000
'''

nerTagger = CoreNLPNERTagger(url='http://localhost:9000')
inputText = 'Your email id is vpoojaridooj@avaya.com'
entities = nerTagger.tag(inputText.split())
print(entities)
for entity in entities:
	if entity[1] == 'CITY':
		print('The city is : ' + entity[0])
	if entity[1] == 'DURATION':
		print('The duration is : ' + entity[0])
	if entity[1] == 'ORGANIZATION':
		print('The organization is : ' + entity[0])
