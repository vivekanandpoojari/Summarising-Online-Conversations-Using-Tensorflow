# Summarising-Online-Conversations-Using-Tensorflow
Summarise online conversations using tensorflow and sci-kit learn

This python script is used to summarize the online conversations.

We have approached summarization as a classification problem. 
We rank the sentences by using normalized scores for various features.
The features which are extracted from each sentence are :

a) Sentence length (longer sentences have more information)
b) Keywords in a sentence e.g. nouns, adjectives, proper nouns
c) Sentence sentiment (negative, postive, neutral)

Each of these scores are normalised and fed into machine learning classification algorithm. 

We have two implementations one in tensorflow and one in scikit.
Both take input in the form of a text file and output a text file.

Contributors :
Vivekanand
Arjun
Sarnava
