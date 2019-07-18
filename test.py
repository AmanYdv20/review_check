import pandas as pd


filename=['amazon','facebook','google','google_map','google_play','messenger','outlook','snapchat','wechat','whatsapp']
data=[]

#data=pd.rfilename=['amazon','facebook','google','google_map','google_play','messenger','outlook','snapchat','wechat','whatsapp']
#data=[]ead_csv('./data/amazon2.csv')

for file in filename:
    count=0
    for i in range(1,5):
        df=pd.read_csv('./data/'+file+str(i)+'.csv')
        df['app_name']=file
        df=df.drop_duplicates()
        count+=df.shape[0]
        data.append(df);
    print(file+'contains',count, 'tweets')

data = pd.concat(data)
data=data[data['text'].apply(lambda x: len(x.split(' ')) > 2)]

data2=[]
for i in range(1,5):
    df=pd.read_csv('./lemmetized_data/output_'+str(i)+'.csv')
    data2.append(df)

data2 = pd.concat(data2)
data2=data2.drop(['Unnamed: 0'],axis=1)

data2.columns=['tweet-id2','timestamp2','like','replies','re-tweets','comment']

final_data=pd.merge(data, data2, left_on='tweet-id', right_on='tweet-id2', how='left').drop('tweet-id2', axis=1)
final_data=final_data.dropna()
final_data=final_data.drop(['like','replies_x','retweets','timestamp2','likes','replies_y','re-tweets'],axis=1)
final_data.to_csv('test_data.csv',index=False)

count=pd.read_csv('final_classification.csv')

snapchat=count[count['app_name']=='snapchat']
snapchat=snapchat[snapchat['bug_report']==1]

def createChunk(df,app_name,start,end):
    df=df[df['app_name']==app_name]
    df=df[df['bug_report']==1]
    df=df[(df['timestamp']>=start) & (df['timestamp']<=end)]
    df.to_csv(app_name+start+'_'+end+'.csv',index=False)
    
createChunk(count,'outlook','2019-04-22','2019-05-31')
#snapchat=snapchat[(snapchat['timestamp']>='2019-03-10') & (snapchat['timestamp']<'2019-03-20')]