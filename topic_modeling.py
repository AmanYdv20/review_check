''''''

import pandas as pd
import random
from pprint import pprint
import gensim
from gensim.models import CoherenceModel
from pre_processing import preprocessing
from check_stemer import tokenize
import re

import pickle
from finding_corpus import findCorpus

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

'''data=[]
for i in range(1,5):
    df=pd.read_csv('./lemmetized_data/output_'+str(i)+'.csv')
    data.append(df)

data = pd.concat(data)
data=data.drop(['Unnamed: 0'],axis=1)'''

#all the code below it is for reviews
data=[]
app_name=['amazon','facebook','google','googleplay','maps','messenger','outlook','snapchat','wechat','whatsapp']
month=['march','april','may']

for mon in month:
    for name in app_name:
        df=pd.read_csv('./detailed_google_reviews/'+name+'_'+mon+'.csv')
        data.append(df)
        
app_name2=['amazon','googleplay','messenger','snapchat','wechat','whatsapp']       
 
for name in app_name2:
    df=pd.read_csv('./detailed_google_reviews/'+name+'_'+'feb.csv')
    data.append(df)

data = pd.concat(data)

data=pd.read_csv('./bug_report_reviews_data/google_music_reviews/Google_Music2019-02-01_2019-03-23.csv')

pre=preprocessing(data)
data['text']=data['text'].apply(tokenize)


corpus_class=findCorpus(data)
corpus=corpus_class.corpus
id2word=corpus_class.id2word
final_data=corpus_class.final_data

# View
print(corpus[:1])

[[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]

#model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=final_data, start=10, limit=50, step=5)

model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=final_data, start=5, limit=25, step=5)

limit=25; start=5; step=5;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()

ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=20, id2word=id2word)
pprint(ldamallet.show_topics(formatted=False))

result=ldamallet.print_topic(8,topn=10)

topics=ldamallet.print_topics(num_topics=20, num_words=10)
#print(topics[0][1])
all_st=[]
all_wt=[]
for topic in topics:
    lst_st=[]
    lst_wt=[]
    s=topic[1]
    val=s.split('+')
    for i in val:
        num,st=i.split('*')
        #print(st)
        st = re.sub('[""]', '', st)
        lst_st.append(str(st.strip()))
        print(lst_st)
        lst_wt.append(float(num))
    all_st.append(lst_st)
    all_wt.append(lst_wt)
    
d={'Keywords_reviews': all_st, 'weights_reviews': all_wt}
#d={'Keywords_tweets': all_st, 'weights_tweets': all_wt}


df1=pd.read_csv('google_music_version_1st.csv')
df2=pd.DataFrame(d)
d=[]
d.append(df1)
d.append(df2)

d=pd.concat(d,axis=1)

d.to_csv('google_music_version_1st.csv',index=False)


#df.to_csv('snapchat_version_1st.csv',index=False)

df.resize((20,4))

df['reviews_keywords']=all_st
df['reviews_weights']=all_wt
    
d={'Keywords_tweets': all_st, 'weights_tweets': all_wt}
d=pd.DataFrame(d)

d.to_csv('google_music_version_1st.csv',index=False)

dt=pd.DataFrame(d)
dt.to_csv('google_version_reviews_4th.csv',index=False)

s=topics[0][1]

val=s.split('+')

for i in val:
    num,st=i.split('*')
    #print(type(float(num)))
    print(st.strip())
    #print(num, st)

filename = 'Snapchat_version_version_1st.pickle'
outfile = open(filename,'wb')
pickle.dump(topics,outfile)
outfile.close()



coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=final_data, dictionary=id2word, coherence='c_v')
coherence_ldamallet = coherence_model_ldamallet.get_coherence()
print('\nCoherence Score: ', coherence_ldamallet)

print(ldamallet[corpus[10]])

def find_topics(data):
    train_vecs = []
    for i in range(len(df)):
        print('executing tweet number', i)
        #top_topics = lda_train.get_document_topics(train_corpus[i], minimum_probability=0.0)
        top_topics = ldamallet[data[i]]
        topic_vec = [top_topics[i][1] for i in range(20)]
        #topic_vec.extend([rev_train.iloc[i].real_counts]) # counts of reviews for restaurant
        #topic_vec.extend([len(rev_train.iloc[i].text)]) # length review
        train_vecs.append(topic_vec)
    return train_vecs
    
#this is code section for tweet data...Will have to restructure after ending the project

'''df=pd.read_csv('labelled_tweets.csv', encoding="latin-1")
df=df.drop(['Unnamed: 7'],axis=1)
df=df.dropna()
df=df[df['Bug_report'].apply(lambda x: str(x).isdigit())]
df.Bug_report = pd.to_numeric(df.Bug_report, errors='coerce')
df=df.reset_index(drop=True)
pre=preprocessing(df)
df=pre.data
df['text']=df['text'].apply(tokenize)
df.to_csv('classifier_final.csv')
df=pd.read_csv('classifier_final.csv')
df=df[:20]
corpus_class=findCorpus(df)
corpus2=corpus_class.corpus
id2word2=corpus_class.id2word
final_data2=corpus_class.final_data
'''

'''df=pd.read_csv('seed.csv')
#df=df.drop(['Unnamed: 7'],axis=1)
df=df.dropna()
df=df[df['Bug_report'].apply(lambda x: str(x).isdigit())]
df.Bug_report = pd.to_numeric(df.Bug_report, errors='coerce')
df=df.reset_index(drop=True)
pre=preprocessing(df)
df=pre.data
df['text']=df['text'].apply(tokenize)
df.to_csv('seed.csv')
df=pd.read_csv('seed.csv')



#df=df[:20]
corpus_class=findCorpus(df)
corpus2=corpus_class.corpus
id2word2=corpus_class.id2word
final_data2=corpus_class.final_data
find_topics(corpus2)

#ldamallet2 = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus2, num_topics=20, id2word=id2word2)
#pprint(ldamallet2.show_topics(formatted=False))
#ldamallet2.print_topics(num_topics=20, num_words=10)

train_vecs = []
for i in range(len(df)):
    print('executing tweet number', i)
    #top_topics = lda_train.get_document_topics(train_corpus[i], minimum_probability=0.0)
    top_topics = ldamallet[corpus2[i]]
    topic_vec = [top_topics[i][1] for i in range(20)]
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
infile.close()'''
