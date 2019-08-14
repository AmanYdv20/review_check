import pandas as pd
import random
from pprint import pprint
import gensim
from gensim.models import CoherenceModel
from pre_processing import preprocessing
from check_stemer import tokenize
import re
import time

import pickle
from finding_corpus import findCorpus

#%matplotlib inline

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
#%matplotlib inline

random.seed(1000)
mallet_path = './mallet-2.0.8/bin/mallet' # update this path

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

df1=pd.read_csv('./bug_report_tweets_data/whatsapp_tweets_data/whatsapp2019-02-01_2019-02-28.csv')

pre=preprocessing(df1)
df1['text']=df1['text'].apply(tokenize)

df2=pd.read_csv('./bug_report_reviews_data/Whatsapp_reviews/Whatsapp2019-02-01_2019-02-28.csv')
df1=df1.reset_index()
df2=df2.reset_index()
df=[]
df.append(df1)
df.append(df2)
df=pd.concat(df)
df=df.drop(['Bug_report','Unnamed: 0','appTitle','app_name','bug_report','original_text','comment','date','id','index','score','timestamp','userName','tweet-id'],axis=1)
corpus_class=findCorpus(df)
corpus=corpus_class.corpus
id2word=corpus_class.id2word
final_data=corpus_class.final_data

print(corpus[:1])

[[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]

model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=final_data, start=5, limit=45, step=5)

limit=45; start=5; step=5;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()

ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=35, id2word=id2word)
pprint(ldamallet.show_topics(formatted=False))

result=ldamallet[corpus[0]]


corpus_class_tweets=findCorpus(df1)
corpus_tweets=corpus_class_tweets.corpus

result=ldamallet[corpus_tweets[0]]
execu_no=0
act_dist_vc=[]
s=time.time()
for corpus_num in corpus_tweets:
    result=ldamallet[corpus_num]
    dist_vc=[]
    for i in result:
        dist_vc.append(i[1])
    print(execu_no)
    execu_no=execu_no+1
    act_dist_vc.append(dist_vc)
f=time.time()
print((f-s)/60,"minutes")

print(act_dist_vc)
act_dis_ser=pd.Series(act_dist_vc)
df1['topic_distribution']=act_dis_ser
df1.to_csv('whatsapp_version_1st.csv',index=False)

corpus_class_reviews=findCorpus(df2)
corpus_reviews=corpus_class_reviews.corpus

act_dist_reviews_vc=[]
execu_no=0
s=time.time()
for corpus_num in corpus_reviews:
    result=ldamallet[corpus_num]
    dist_vc=[]
    for i in result:
        dist_vc.append(i[1])
    #print(dist_vc)
    print(execu_no)
    execu_no=execu_no+1
    act_dist_reviews_vc.append(dist_vc)
f=time.time()
print((f-s)/60,"minutes")



print(act_dist_reviews_vc)
act_dis_reviews_ser=pd.Series(act_dist_reviews_vc)
df2['topic_distribution']=act_dis_reviews_ser
df2.to_csv('whatsapp_reviews_version_1st.csv',index=False)


    



