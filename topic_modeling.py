import re
import pandas as pd
import numpy as np
import random
from pprint import pprint
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import spacy
from pre_processing import preprocessing
from check_stemer import tokenize
# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import pickle
#%matplotlib inline

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

random.seed(1000)

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu','whatsit','atUse'])

mallet_path = './mallet-2.0.8/bin/mallet' # update this path

# Initialize spacy 'en' model
nlp = spacy.load('en', disable=['parser', 'ner'])
    
def replaceImage(text):
    """ Replaces url address with "url" """
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(pictwiter[^\s]+))','',text)
    return text

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

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
    


data=[]
for i in range(1,7):
    df=pd.read_csv('./lemmetized_data/output_'+str(i)+'.csv')
    data.append(df)

data = pd.concat(data)

data=pre_steps(data)

data_words = list(sent_to_words(data))

bigram = gensim.models.Phrases(data_words, min_count=5, threshold=50) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=50)
# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)
    # Remove Stop Words
data_words_nostops = remove_stopwords(data_words)
# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

data_lemmatized = data_words_bigrams

print(trigram_mod[bigram_mod[data_words[0]]])

print(data_lemmatized[:1])
# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
#this is the train_corpus
corpus = [id2word.doc2bow(text) for text in texts]

#*******************************************************************************************
#lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
#                                           id2word=id2word,
#                                           num_topics=25, 
#                                           random_state=100,
#                                           update_every=1,
#                                           chunksize=100,
#                                           passes=10,
#                                           alpha='auto',
#                                           per_word_topics=True)
#*******************************************************************************************
# View
print(corpus[:1])

[[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]

ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=25, id2word=id2word)
pprint(ldamallet.show_topics(formatted=False))

coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_ldamallet = coherence_model_ldamallet.get_coherence()
print('\nCoherence Score: ', coherence_ldamallet)

print(ldamallet[corpus[10]])
#************************************************************
#So,until here, all parts related to finding the topics for the model is completed

#model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_lemmatized, start=5, limit=55, step=5)

#limit=55; start=5; step=5;
#x = range(start, limit, step)
#plt.plot(x, coherence_values)
#plt.xlabel("Num Topics")
#plt.ylabel("Coherence score")
#plt.legend(("coherence_values"), loc='best')
#plt.show()

#for m, cv in zip(x, coherence_values):
#    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))

#*************************************************************

def make_bigrams(texts):
    return [bigram_mod2[doc] for doc in texts]

def make_trigrams2(texts):
    return [trigram_mod2[bigram_mod2[doc]] for doc in texts]

df=pd.read_csv('random_data_7000.csv')
df=df.drop(['Unnamed: 7','Unnamed: 8','Unnamed: 9'],axis=1)
df=df.dropna()
df=df[df['Bug_report'].apply(lambda x: str(x).isdigit())]
df.Bug_report = pd.to_numeric(df.Bug_report, errors='coerce')
df.reset_index()
pre=preprocessing(df)
df=pre.data
df['text']=df['text'].apply(tokenize)
df.to_csv('classifier_final.csv')
df=pd.read_csv('classifier_final.csv')
df=pre_steps(df)


data_words2 = list(sent_to_words(df))


bigram2 = gensim.models.Phrases(data_words2, min_count=1, threshold=1) # higher threshold fewer phrases.
trigram2 = gensim.models.Phrases(bigram2[data_words2], threshold=3)
# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod2 = gensim.models.phrases.Phraser(bigram2)
trigram_mod2 = gensim.models.phrases.Phraser(trigram2)
    # Remove Stop Words
data_words_nostops2 = remove_stopwords(data_words2)
# Form Bigrams
data_words_bigrams2 = make_bigrams(data_words_nostops2)

data_lemmatized2 = data_words_bigrams2

print(trigram_mod2[bigram_mod2[data_words2[0]]])

print(data_lemmatized2[:1])
# Create Dictionary
id2word2 = corpora.Dictionary(data_lemmatized2)

# Create Corpus
texts2 = data_lemmatized2

# Term Document Frequency
#this is the train_corpus
corpus2 = [id2word2.doc2bow(text) for text in texts2]
# View
print(corpus2[:1])

print(ldamallet[corpus2[1]])

train_vecs = []
for i in range(len(df)):
    print('executing tweet number', i)
    #top_topics = lda_train.get_document_topics(train_corpus[i], minimum_probability=0.0)
    top_topics = ldamallet[corpus2[i]]
    topic_vec = [top_topics[i][1] for i in range(25)]
    #topic_vec.extend([rev_train.iloc[i].real_counts]) # counts of reviews for restaurant
    #topic_vec.extend([len(rev_train.iloc[i].text)]) # length review
    train_vecs.append(topic_vec)

#for storing the assigned topic results in pickle format
filename = 'assigned_topics.pickle'
outfile = open(filename,'wb')
pickle.dump(train_vecs,outfile)
outfile.close()

#for extracting the data out of pickle file
infile = open(filename,'rb')
test_pic = pickle.load(infile)
infile.close()




