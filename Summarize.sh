clear

rm -rf "/Users/Sarnava/Documents/temp/SummarizeTranscript.py"

cp "/Users/Sarnava/Data/hackathon/Hackathon_2018/SummarizeTranscript.py" "/Users/Sarnava/Documents/temp"

cd /Users/Sarnava/Data/hackathon/stanford-corenlp-full-2018-02-27
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer

 
