from rake_nltk import Rake
r = Rake()
r.extract_keywords_from_text('is your lan cable connected?')
keywords = r.get_ranked_phrases()
for keyword in keywords:
	print(keyword)

