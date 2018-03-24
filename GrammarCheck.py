import grammar_check
string = 'I can see the light is blinking, but still my wifi connection is not working. I do not remember. He am playing. my name is Sarn'
print(string)

repls = (' I ', ' He ' ), (' my ', ' his '), (' me ',' him '), (' mine ',' his ') , ('I ', 'He ')
repls_1 = (' He ', ' Customer ' ), (' his ', ' Customer\'s '), (' him ',' Customer\'s '),  ('He ', 'Customer ')
string = reduce(lambda a, kv: a.replace(*kv), repls, string)

print("After replacement :- " + string)



tool= grammar_check.LanguageTool('en-GB')
matches = tool.check(string)
len(matches)
#print(string)
string = grammar_check.correct(string,matches)
string = reduce(lambda a, kv: a.replace(*kv), repls_1, string)
print (string)
