import pandas as pd
import datetime
from matplotlib import pyplot

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
count=count[count['bug_report']==1]

count.reset_index()
count=count.drop_duplicates()

count.groupby(['app_name']).size()

#count.to_csv('bug_report_data_tweets.csv',index=False)
d=pd.read_csv('bug_report_data_tweets.csv')
d.groupby(['app_name']).agg(['count'])

snapchat=count[count['app_name']=='messenger']
snapchat=snapchat[snapchat['bug_report']==1]

def createChunk(df,app_name,start,end):
    df=df[df['app_name']==app_name]
    df=df[df['bug_report']==1]
    df=df[(df['timestamp']>=start) & (df['timestamp']<=end)]
    df.to_csv(app_name+start+'_'+end+'.csv',index=False)
    
createChunk(count,'google_play','2019-03-24','2019-05-31')
#snapchat=snapchat[(snapchat['timestamp']>='2019-03-10') & (snapchat['timestamp']<'2019-03-20')]


#testing code start from here
#df=pd.read_json('all.json')

#code for separating bug_report

count=pd.read_csv('bug_report_data_reviews.csv')
#count=count[count['date']>='February 18,2019']

def createChunk(df,app_name,start,end):
    df=df[df['appTitle']==app_name]
    df=df[df['Bug_report']==1]
    df=df[(df['date']>=start) & (df['date']<=end)]
    df.to_csv(app_name+start+'_'+end+'.csv',index=False)
    
#count=count[count['appTitle']=='Amazon']
count['date'] = pd.to_datetime(count['date'])
createChunk(count,'Maps','2019-02-01', '2019-03-31')
print(type(count['date'][21]))


#below code is for finding tweets and retweets
d=pd.read_csv('bug_report_data_tweets.csv')
data.columns=['tweet-id2','timestamp','likes','replies','retweets','text','app_name']
d=d.dropna()
d[['tweet-id']]=d[['tweet-id']].apply(pd.to_numeric)
d=pd.merge(d, data, left_on='tweet-id', right_on='tweet-id2', how='left').drop('tweet-id2', axis=1)

d=d.drop(['timestamp_y','text_y','app_name_y'],axis=1)
d.columns=['tweet-id','timestamp','text','app_name','comment','bug_report','likes','replies','retweets']
d.to_csv('bug_report_tweets_data.csv',index=False)

plot_data=d.groupby('timestamp').count()
plot_data['timestamp']=plot_data.index
plot_data['timestamp']=pd.to_datetime(plot_data['timestamp'])
data1=plot_data['timestamp'][0]
plot_data['Date'] = [dt.datetime.date(d) for d in plot_data['timestamp']]
plot_data=plot_data.groupby('Date').count()
plot_data.plot()
pyplot.show()

df=plot_.groupby('Name').resample('W-Mon', on='Date').sum().reset_index().sort_values(by='Date')


d["timestamp"] = pd.to_datetime(d["timestamp"])
d['date_minus_time'] = d["timestamp"].apply( lambda d : datetime.datetime(year=d.year, month=d.month, day=d.day))	
d.set_index(d["date_minus_time"],inplace=True)

df=d['likes'].resample('W', how='sum')
df.plot()
pyplot.show()
