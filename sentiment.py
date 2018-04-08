'''
Navigate to standford core nlp folder and execute following command.
java -mx5g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -timeout 10000
'''

from pycorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP('http://localhost:9000')
res = nlp.annotate("I am not satisfied with your service. I want to discontinue your service. Yes, my modem is working.",
                   properties={
                       'annotators': 'sentiment',
                       'outputFormat': 'json',
                       'timeout': 1000,
                   })
for s in res["sentences"]:
	print("%d: '%s': %s %s" % (
        s["index"],
        " ".join([t["word"] for t in s["tokens"]]),
        s["sentimentValue"], s["sentiment"]))

output = nlp.annotate("Lets meet at Milton", 
					properties={
						'annotators': 'tokenize,ssplit,pos,depparse,parse',
  						'outputFormat': 'json'
  					})

