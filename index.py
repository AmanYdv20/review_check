import pandas as pd
from pre_processing import preprocessing
from check_stemer import tokenize

data=pd.read_csv('facebook.csv')
data=data.drop_duplicates()
pre=preprocessing(data)
data=pre.data
#for removing tweets having length less than 3
data=data[data['text'].apply(lambda x: len(x.split(' ')) > 3)]
print(data)
data['text']=data['text'].apply(tokenize)
data.to_csv('facebook_output_onlylemm.csv')
