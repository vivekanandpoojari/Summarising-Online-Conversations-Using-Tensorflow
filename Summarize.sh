clear

rm -rf "/Users/csdkpune/Documents/temp/Hackathon_2018/SummarizeTranscript.py"

cp "/Users/csdkpune/Documents/temp/Hackathon_2018/main.py" "/Users/csdkpune/Documents/temp/SummarizeTranscript.py"

cd /Users/csdkpune/Documents/Hackathon/stanford-corenlp-full-2018-02-27
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer

 