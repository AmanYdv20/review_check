import re
from functools import partial
from collections import Counter
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import string
from spellchecker import SpellChecker

spell=SpellChecker()
stoplist = stopwords.words('english')
my_stopwords = "multiexclamation multiquestion multistop url atuser st rd nd th am pm" # my extra stopwords
stoplist = stoplist + my_stopwords.split()
lemmatizer = WordNetLemmatizer() # set lemmatizer
stemmer = PorterStemmer() # set stemmer
i=0

def addCapTag(word):
    """ Finds a word with at least 3 characters capitalized and adds the tag ALL_CAPS_ """
    if(len(re.findall("[A-Z]{3,}", word))):
        word = word.replace('\\', '' )
        transformed = re.sub("[A-Z]{3,}", "ALL_CAPS_"+word, word)
        return transformed
    else:
        return word
    

def replaceElongated(word):
    """ Replaces an elongated word with its basic form, unless the word exists in the lexicon """

    repeat_regexp = re.compile(r'(\w*)(\w)\2(\w*)')
    repl = r'\1\2\3'
    if wordnet.synsets(word):
        return word
    repl_word = repeat_regexp.sub(repl, word)
    if repl_word != word:      
        return replaceElongated(repl_word)
    else:       
        return repl_word

def tokenize(text):
    global x
    final_tokens=[]
    tokens = nltk.word_tokenize(text)
    
    #tokens = replaceNegations(tokens) # Technique 6: finds "not" and antonym for the next word and if found, replaces not and the next word with the antonym
    
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator) # Technique 7: remove punctuation

    tokens = nltk.word_tokenize(text) # it takes a text as an input and provides a list of every token in it
    
### NO POS TAGGING BEGIN (If you don't want to use POS Tagging keep this section uncommented) ###
    
    for w in tokens:

        if (w not in stoplist): # Technique 10: remove stopwords
            #final_word = addCapTag(w) # Technique 8: Finds a word with at least 3 characters capitalized and adds the tag ALL_CAPS_
            final_word = w.lower() # Technique 9: lowercases all characters
            final_word = replaceElongated(final_word) # Technique 11: replaces an elongated word with its basic form, unless the word exists in the lexicon
            if len(final_word)>1:
                final_word = spell.correction(final_word) # Technique 12: correction of spelling errors
            final_word = lemmatizer.lemmatize(final_word) # Technique 14: lemmatizes words
            #final_word = stemmer.stem(final_word) # Technique 15: apply stemming to words
            
            final_tokens.append(final_word)
    
    text=' '.join([i for i in final_tokens])
    print("execution number", x)
    x=x+1
    return text

#sentence="This is not a good examle you are setting among the public!!"
#ans=tokenize(sentence)
