import pandas as pd
from pre_processing import preprocessing
from check_stemer import tokenize
from nltk.util import ngrams
from collections import Counter
from functools import partial
import random
from finding_grams import finding_ngram
from nltk.tokenize import word_tokenize
#import random

random.seed(1000)

filename=['amazon','facebook','google','google_map','google_play','messenger','outlook','snapchat','wechat','whatsapp']
data=[]

#data=pd.rfilename=['amazon','facebook','google','google_map','google_play','messenger','outlook','snapchat','wechat','whatsapp']
#data=[]ead_csv('./data/amazon2.csv')

for file in filename:
    count=0
    for i in range(1,5):
        df=pd.read_csv('./data/'+file+str(i)+'.csv')
        df=df.drop_duplicates()
        count+=df.shape[0]
        data.append(df);
    print(file+'contains',count, 'tweets')

data = pd.concat(data)
#pre=preprocessing(data)
#data=pre.data
data=data.reset_index(drop=True)
data.insert(5, "Bug_report",'')

data2=pd.read_csv(r'random_data_7000.csv')
data2=data2.drop(['Unnamed: 7','Unnamed: 8','Unnamed: 9'],axis=1)

finding_ngram(data,2000,3)


data=data[data['text'].apply(lambda x: len(word_tokenize(x)) > 4)]
random_subset = data.sample(n=7000)

#df=[random_subset,data2]
#df=pd.concat(df)

#df = df.sort_values('tweet-id', ascending=True)
df = df.drop_duplicates(subset='Name', keep='first')

export_csv = random_subset.to_csv (r'randomdata_7000.csv', index = None, header=True)


#data=data[data['text'].apply(lambda x: len(x.split(' ')) > )]
size=data.shape
n=random.randint(1,size[0]-7000)
data=data[n:n+7000]
export_csv = data.to_csv (r'random_data_sequential.csv', index = None, header=True)

data['text']=data['text'].apply(tokenize)
export_csv = data.to_csv (r'random_data.csv', index = None, header=True)

#data=pd.read_csv('facebook.csv')
#data=data.drop_duplicates()
#pre=preprocessing(data)
#data=pre.data
#for removing tweets having length less than 3
#data=data[data['text'].apply(lambda x: len(x.split(' ')) > 3)]
#print(data)
#data['text']=data['text'].apply(tokenize)
#data.to_csv('facebook_output_onlylemm.csv')
