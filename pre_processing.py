#This file is for preprocessing the raw text data and it perfoms various operation like coherence resolution,removing unicode, removing extra exclamtion mark etc.
#if any time you stuck somewhere then you can check it on newfile.log file or write your own log file for more info

import re
import spacy
import neuralcoref
from nltk.corpus import wordnet
import pandas as pd
from changeSlang import replaceAllSlang

import logging
logging.basicConfig(filename="newfile.log", format='%(asctime)s %(message)s', filemode='w')
logger=logging.getLogger()
logger.setLevel(logging.INFO)

#from nltk import word_tokenize
#from nltk.corpus import stopwords
#from num2words import num2words
#from nltk.util import ngrams
nlp=spacy.load('en')
neuralcoref.add_to_pipe(nlp)

def coreference_resolution(text):
    logger.info("Executed successfully") 
    print('execute')
    doc=nlp(text)
    ans=doc._.coref_resolved
    return ans

def removeUnicode(text):
    """ Removes unicode strings like "\u002c" and "x96" """
    text=str(text)
    text = re.sub(r'(\\u[0-9A-Fa-f]+)',r'', text)       
    text = re.sub(r'[^\x00-\x7f]',r'',text)
    return text

def removeNumbers(text):
    """ Removes integers """
    text = ''.join([i for i in text if not i.isdigit()])         
    return text

def replaceAlpha(text):
    """ Removes integers """
    sentence=""
    for i in text:
        if i.isalpha():
            i=i.lower()
        sentence+=i
    return sentence

def replaceURL(text):
    """ Replaces url address with "url" """
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(pic\.[^\s]+))',r'',text)
    text = re.sub(r'#([^\s]+)', r'\1', text)
    return text

def replaceAtUser(text):
    """ Replaces "@user" with "atUser" """
    text = re.sub('@[^\s]+',r'',text)
    return text

def removeHashtagInFrontOfWord(text):
    """ Removes hastag in front of a word """
    text = re.sub(r'#([^\s]+)', r'\1', text)
    return text


def replaceMultiExclamationMark(text):
    """ Replaces repetitions of exlamation marks """
    text = re.sub(r"(\!)\1+", '!', text)
    return text

def replaceMultiQuestionMark(text):
    """ Replaces repetitions of question marks """
    text = re.sub(r"(\?)\1+", '', text)
    return text

def replaceMultiStopMark(text):
    """ Replaces repetitions of stop marks """
    text = re.sub(r"(\.)\1+", '.', text)
    return text

def removeEmoticons(text):
    """ Removes emoticons from text """
    text = re.sub(':\)|;\)|:-\)|\(-:|:-D|=D|:P|xD|X-p|\^\^|:-*|\^\.\^|\^\-\^|\^\_\^|\,-\)|\)-:|:\'\(|:\(|:-\(|:\S|T\.T|\.\_\.|:<|:-\S|:-<|\*\-\*|:O|=O|=\-O|O\.o|XO|O\_O|:-\@|=/|:/|X\-\(|>\.<|>=\(|D:', '', text)
    return text

#class for performing initials functions to the dataset
class preprocessing:
    def __init__(self,df):
        self.data=df
        #self.data['text']=self.data['text'].apply(replaceToLower)
        self.data['text']=self.data['text'].apply(removeUnicode)
        self.data['text']=self.data['text'].apply(replaceURL)
        self.data['text']=self.data['text'].apply(removeHashtagInFrontOfWord)
        self.data['text']=self.data['text'].apply(replaceAtUser)
        self.data['text']=self.data['text'].apply(replaceMultiQuestionMark)
        self.data['text']=self.data['text'].apply(replaceMultiStopMark)
        self.data['text']=self.data['text'].apply(removeEmoticons)
        self.data['text']=self.data['text'].apply(replaceMultiExclamationMark)
        self.data['text']=self.data['text'].apply(removeNumbers)
        self.data['text']=self.data['text'].apply(replaceAlpha)
        self.data['text']=self.data['text'].apply(replaceAllSlang)
        self.data['text']=self.data['text'].apply(coreference_resolution)
        
        
    def print_data(self):
        print(self.data.head())



        
