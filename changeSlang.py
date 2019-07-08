import re
from functools import partial
import nltk
#import pandas as pd
#import re
#from collections import Counter
#from nltk.corpus import wordnet
#from nltk.corpus import stopwords
#from nltk.stem import WordNetLemmatizer
#from nltk.stem.porter import PorterStemmer
""" Creates a dictionary with slangs and their equivalents and replaces them """
with open('slang.txt') as file:
    slang_map = dict(map(str.strip, line.partition('\t')[::2])
    for line in file if line.strip())

slang_words = sorted(slang_map, key=len, reverse=True) # longest first for regex
regex = re.compile(r"\b({})\b".format("|".join(map(re.escape, slang_words))))
replaceSlang = partial(regex.sub, lambda m: slang_map[m.group(1)])

def countSlang(text):
    """ Input: a text, Output: how many slang words and a list of found slangs """
    slangCounter = 0
    slangsFound = []
    tokens = nltk.word_tokenize(text)
    for word in tokens:
        if word in slang_words:
            slangsFound.append(word)
            slangCounter += 1
    return slangCounter, slangsFound

""" Replaces contractions from a string to their equivalents """
contraction_patterns = [ (r'won\'t', 'will not'), (r'can\'t', 'cannot'),(r'cant', 'cannot'), (r'i\'m', 'i am'),(r'\bim\b ', 'i am'), (r'ain\'t', 'is not'), (r'(\w+)\'ll', '\g<1> will'), (r'(\w+)n\'t', '\g<1> not'),
                         (r'(\w+)\'ve', '\g<1> have'), (r'(\w+)\'s', '\g<1> is'), (r'(\w+)\'re', '\g<1> are'), (r'(\w+)\'d', '\g<1> would'), (r'&', 'and'), (r'dammit', 'damn it'), (r'\bdont\b', 'do not'), (r'\bwont\b', 'will not'),(r'\bive\b', 'I have'),(r'\bIve\b', 'I have'),(r'i\'d', 'I would'),(r'I\'m', 'I would')]
def replaceContraction(text):
    patterns = [(re.compile(regex), repl) for (regex, repl) in contraction_patterns]
    for (pattern, repl) in patterns:
        (text, count) = re.subn(pattern, repl, text)
    return text

def replaceAllSlang(text):
    totalSlangs=0
    totalSlangsFound = []
    temp_slangs, temp_slangsFound = countSlang(text)
    totalSlangs += temp_slangs # total slangs for all sentences
    for word in temp_slangsFound:
        totalSlangsFound.append(word) # all the slangs found in all sentences
    
    text = replaceSlang(text) # Technique 2: replaces slang words and abbreviations with their equivalents
    text = replaceContraction(text)
    return text
    
#ans=replaceAllSlang("btw, Ive ab dont to hell")
#print(ans)