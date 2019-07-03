import pandas as pd
from pre_processing import preprocessing
from check_stemer import tokenize
from nltk.util import ngrams
from collections import Counter
from functools import partial
import random

filename=['amazon','facebook','google','google_map','google_play','messenger','outlook','snapchat','wechat','whatsapp']
data=[]

for file in filename:
    count=0
    for i in range(1,5):
        df=pd.read_csv('./data/'+file+str(i)+'.csv')
        df=df.drop_duplicates()
        count+=df.shape[0]
        data.append(df);
    print(file+'contains',count, 'tweets')

data = pd.concat(data)
pre=preprocessing(data)
data=pre.data
data.reset_index()


data=data[data['text'].apply(lambda x: len(x.split(' ')) > 3)]
random_subset = data.sample(n=7000)
export_csv = random_subset.to_csv (r'random_data_7000.csv', index = None, header=True)


#data=data[data['text'].apply(lambda x: len(x.split(' ')) > )]
size=data.shape
n=random.randint(1,size[0]-7000)
data=data[n:n+7000]
export_csv = data.to_csv (r'random_data_sequential.csv', index = None, header=True)

data['text']=data['text'].apply(tokenize)
export_csv = data.to_csv (r'random_data.csv', index = None, header=True)
text=''
for i in data['text']:
    text+=i
    text+=' '
tokenized = text.split()
esBigrams = ngrams(tokenized, 2)
esBigramFreq = Counter(esBigrams)
most_frequent=esBigramFreq.most_common(4500)
print(most_frequent)

f = open( 'bigrams.txt', 'w' )
f.write( 'most frequent bigrams= ' + repr(most_frequent) + '\n' )
f.close()

triBigrams = ngrams(tokenized, 3)
triBigramFreq = Counter(triBigrams)
most_frequent=triBigramFreq.most_common(2000)
print(most_frequent)

f = open( 'trigrams.txt', 'w' )
f.write( 'most frequent bigrams= ' + repr(most_frequent) + '\n' )
f.close()

quadBigrams = ngrams(tokenized, 4)
quadBigramFreq = Counter(quadBigrams)
most_frequent=quadBigramFreq.most_common(750)
print(most_frequent)

f = open( 'quadgrams.txt', 'w' )
f.write( 'most frequent bigrams= ' + repr(most_frequent) + '\n' )
f.close()
#data=pd.read_csv('facebook.csv')
#data=data.drop_duplicates()
#pre=preprocessing(data)
#data=pre.data
#for removing tweets having length less than 3
#data=data[data['text'].apply(lambda x: len(x.split(' ')) > 3)]
#print(data)
#data['text']=data['text'].apply(tokenize)
#data.to_csv('facebook_output_onlylemm.csv')
