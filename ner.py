from rake_nltk import Rake

import nltk
import grammar_check
from nltk.tokenize import word_tokenize

r = Rake()
myText = "When will my phone be repaired ? "
r.extract_keywords_from_text(myText)
print(r.get_ranked_phrases())
answer = "It is being shipped right now and It will reach in 5 minutes"

text = ''
min1 = -1
max1 = 0
j = 0
for i in r.get_ranked_phrases():
    j = myText.index(i)
    if (min1 == -1):
        min1 = j
    if (j < min1):
        min1 = j
print(myText[min1:])
            
statement = myText[min1:] + " : " + answer
print(statement)

'''
myText = "Is your mobile number regsitered ? "
r.extract_keywords_from_text(myText)
print(r.get_ranked_phrases())


#text = word_tokenize(myText)
#print(nltk.pos_tag(text))



sent = nltk.corpus.treebank.tagged_sents()[22]
print(sent)

print(nltk.ne_chunk(sent, binary=True)) 


text = 'yes wife at home today ?'

for i in r.get_ranked_phrases():
    text += ' ' + i

text += mytext
print(text)


tool= grammar_check.LanguageTool('en-GB')
matches = tool.check(text)
#print(len(matches))
#print(string)
string = grammar_check.correct(text, matches)

#print(matches)
#string = reduce(lambda a, kv: a.replace(*kv), repls_1, string)
#print (string)

'''
