#This is a general file to import all the data of tweets. Apart from that there is no use of this file
import pandas as pd
from pre_processing import preprocessing
import random
from finding_grams import finding_ngram

random.seed(1000)

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
data=data.reset_index()
data.insert(5, "Bug_report",'')

data2=pd.read_csv(r'random_data_7000.csv')
data2=data2.drop(['Unnamed: 7','Unnamed: 8','Unnamed: 9'],axis=1)

finding_ngram(data,2000,3)

