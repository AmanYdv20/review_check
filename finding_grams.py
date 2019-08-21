''' This file is to create the trigrams and quadgrams according the condition you will dercribed in below code
You have to manually specify the the file names each time you will extract n-grams
'''

import pandas as pd
from nltk.util import ngrams
from collections import Counter
from finding_corpus import sent_to_words
from finding_corpus import pre_steps
#from finding_corpus import remove_stopwords
from pre_processing import preprocessing
from check_stemer import tokenize
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

extra_words = ['still','not','into','in','this','did','doing','because','until','while','having']
for word in extra_words:
    if word in stop_words:
        stop_words.remove(word)

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

'''function for finding the ngrams'''
def finding_ngram(df,min_count,n=2):
    '''df: dataframe with text column'''
    text=''
    for i in df['text']:
        text+=i+' '
    tokenized=text.split()
    esngrams = ngrams(tokenized, n)
    esngramFreq = Counter(esngrams)
    most_frequent=esngramFreq.most_common(min_count)
    print(most_frequent)
    f = open( str(n)+'grams.txt', 'w' )
    f.write( 'most frequent bigrams= ' + repr(most_frequent) + '\n' )
    f.close()
    
#specify the filename here for finding the trigrams and quadgrams
data=pd.read_csv('./bug_report_tweets_data/google_tweets_data/google2019-02-01_2019-02-28.csv')

data=pd.read_csv('./bug_report_reviews_data/google_reviews/Google2019-02-01_2019-02-28.csv')

pre=preprocessing(data)
data['text']=data['text'].apply(tokenize)

data=pre_steps(data)
data_words=list(sent_to_words(data))
data_words_nostops=remove_stopwords(data_words)

all_word=[]
for i in data_words_nostops:
    all_word+=i

#tokenized=text.split()
esngrams=ngrams(all_word, 4)
esngramFreq = Counter(esngrams)
most_frequent=esngramFreq.most_common(2500)
#print(most_frequent[0][1])

sum_val=0

for i in most_frequent:
    sum_val+=i[1]

sum_val=sum_val*0.5
print(sum_val)

df=pd.DataFrame(most_frequent)
df.columns=['trigrams_tweets','frequency']
df=df[(df['frequency']<int(sum_val))&(df['frequency']>1)]

df2=pd.DataFrame(most_frequent)
df2.columns=['trigrams_reviews','frequency_reviews']
df2=df2[(df2['frequency_reviews']<int(sum_val))&(df2['frequency_reviews']>3)]


df=pd.DataFrame(most_frequent)
df.columns=['quadgrams_tweets','frequency']
df=df[(df['frequency']<int(sum_val))&(df['frequency']>1)]

df2=pd.DataFrame(most_frequent)
df2.columns=['quadgrams_reviews','frequency_reviews']
df2=df2[(df2['frequency_reviews']<int(sum_val))&(df2['frequency_reviews']>4)]

d=[]
d.append(df)
d.append(df2)

d=pd.concat(d,axis=1)

#specify the desired filename you want to save the quadgrams
d.to_csv('google_trigram_version_4th.csv',index=False)
d.to_csv('google_quadgram_version_1st.csv',index=False)





    



    
