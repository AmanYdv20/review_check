import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import gensim
from gensim.models import LdaModel
from gensim.models import CoherenceModel
from pre_processing import preprocessing
from check_stemer import tokenize
from gensim import models, corpora, similarities
import time
from nltk import FreqDist
from scipy.stats import entropy
from finding_corpus import findCorpus
import ast

def jensen_shannon(query, matrix):
    """
    This function implements a Jensen-Shannon similarity
    between the input query (an LDA topic distribution for a document)
    and the entire corpus of topic distributions.
    It returns an array of length M where M is the number of documents in the corpus
    """
    # lets keep with the p,q notation above
    p = query[None,:].T # take transpose
    q = matrix.T # transpose matrix
    m = 0.5*(p + q)
    return np.sqrt(0.5*(entropy(p,m) + entropy(q,m)))

def get_most_similar_documents(query,matrix,k=5):
    """
    This function implements the Jensen-Shannon distance above
    and retruns the top k indices of the smallest jensen shannon distances
    """
    sims = jensen_shannon(query,matrix) # list of jensen shannon distances
    sims_idx=sims.argsort()[:k]# the top k positional index of the smallest Jensen Shannon distances
    sims_val=np.sort(sims)
    return sims_idx,sims_val[:k]

def change(text):
    return ast.literal_eval(text)

df1=pd.read_csv('./whatsapp_topic_distribution/whatsapp_tweets_version_1st.csv')
df1=df1.dropna()

df2=pd.read_csv('./whatsapp_topic_distribution/whatsapp_reviews_version_1st.csv')
df2=df2.drop(['index','Unnamed: 0','bug_report','userName','Unnamed: 12','Unnamed: 13'],axis=1)
df2=df2.dropna()
df2['topic_distribution']=df2['topic_distribution'].apply(change)
df2=df2.reset_index()

y=[]
for j in df2['topic_distribution']:
    y.append(j)
    
    
df1['topic_distribution']=df1['topic_distribution'].apply(change)
df1=df1[df1['topic_distribution'].apply(lambda x:max(x)>=0.03)]
rows=np.random.choice(df1.index.values,100)

#sampled_df=df1.ix[rows]
sampled_df=df1
sampled_df=sampled_df.reset_index()
sampled_df=sampled_df.drop(['index'],axis=1)
#x=sampled_df['topic_distribution'][10]

#most_sim_ids,val = get_most_similar_documents(np.array(x),np.array(y))
#mask = df2.index.isin(most_sim_ids)
#similiar_reviews=df2[mask]

tweets=[]
tweets_dist=[]
reviews=[]
reviews_dist=[]
score=[]

for x in sampled_df['topic_distribution']:
    most_sim_ids,val=get_most_similar_documents(np.array(x),np.array(y))
    mask=df2.index.isin(most_sim_ids)
    similiar_reviews=df2[mask]
    reviews_temp=[]
    for i in similiar_reviews['text']:
        reviews.append(i)
    for j in similiar_reviews['topic_distribution']:
        reviews_dist.append(j)
    score=score+list(val)
    for k in range(0,5):
        tweets_dist.append(x)
    
for tweet in sampled_df['text']:
    for k in range(0,5):
        tweets.append(tweet)

score=[str(i) for i in score]

data=pd.DataFrame({'tweets':tweets,'tweets_distribution': tweets_dist,'reviews': reviews,'reviews_dist':reviews_dist})
data['gsd_score']=score

data.to_csv('whatsapp_combined_version_1st.csv',index=False)


#only for some testing
d=pd.read_csv('./combined_topic_distribution/snapchat_combined_distribution/snapchat_combined_version_1st.csv')
rows=np.random.choice(d.index.values,26)
sampled_d=d.ix[rows]
sampled_d.to_csv('human_eval_data.csv',index=False)

mn_f=pd.read_csv('human_eval_data.csv')
d=pd.read_csv('./combined_topic_distribution/snapchat_combined_distribution/snapchat_combined_version_4th.csv')
rows=np.random.choice(d.index.values,26)
sampled_d=d.ix[rows]
data=[]
data.append(mn_f)
data.append(sampled_d)
data=pd.concat(data)
data.to_csv('human_eval_data.csv',index=False)

    
    





