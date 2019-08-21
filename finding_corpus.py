#This file is responsible for all the preprocessing before topic modelling of the text document.
#It convert text to id2word and combining the bigram and trigrams to the text
#the output of this file can be seen from the topic_modelling file
#Please aware of the fact the code is considering the ranw text data in the column name "text", so please do not forgot it otherwise you will get an error.

import re
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
import spacy
from nltk.corpus import stopwords

#importing the english model of spacy library 
nlp = spacy.load('en', disable=['parser', 'ner'])

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu','whatsit','atuser','url'])

extra_words = ['still','not','soon','into','in','have','been','doing','because','until','having']
for word in extra_words:
    if word in stop_words:
        stop_words.remove(word)

#replaces all the hyper link if any remaining after the preprocessing of data
def replaceImage(text):
    """ Replaces url address with "url" """
    text=str(text)
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(pictwiter[^\s]+))','',text)
    return text

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

#lemmetizing the text
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

#doing some preprocssing steps to confirm if everything is fine and remove any mistake if any.
def pre_steps(data):
    data['text']=data['text'].apply(replaceImage)
    data = data.text.values.tolist()
    data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]
    # Remove new line characters
    data = [re.sub('\s+', ' ', sent) for sent in data]
    # Remove distracting single quotes
    data = [re.sub("\'", "", sent) for sent in data]
    # Build the bigram and trigram models
    # See trigram example
    
    return data

#class for finding the corpus for the given data
class findCorpus:
    def __init__(self,data):
        self.data=data
        self.data=pre_steps(self.data)
        self.data_words=list(sent_to_words(self.data))
        self.bigram = gensim.models.Phrases(self.data_words, min_count=10, threshold=30) # higher threshold fewer phrases.
        self.trigram = gensim.models.Phrases(self.bigram[self.data_words], threshold=30)
        self.bigram_mod = gensim.models.phrases.Phraser(self.bigram)
        self.trigram_mod = gensim.models.phrases.Phraser(self.trigram)
        self.data_words_nostops=remove_stopwords(self.data_words)
        self.data_words_bigrams = self.make_bigrams()
        self.data_words_trigrams = self.make_trigrams()
        self.final_data=lemmatization(self.data_words_bigrams)
        self.id2word = corpora.Dictionary(self.final_data)
        self.corpus = [self.id2word.doc2bow(text) for text in self.final_data]
    
    def make_bigrams(self):
        return [self.bigram_mod[doc] for doc in self.data_words_nostops]
    
    def make_trigrams(self):
        return [self.trigram_mod[self.bigram_mod[doc]] for doc in self.data_words_nostops]
    
    