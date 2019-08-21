import pandas as pd
import datetime
import matplotlib.pyplot as plt

df1=pd.read_csv('bug_report_tweets_data.csv')
df2=pd.read_csv('bug_report_data_reviews.csv')	


df1=df1[df1['app_name']=='google']
df2=df2[df2['appTitle']=='Google']

df2=df2.drop(['index','Unnamed: 0','bug_report','userName','Unnamed: 12','Unnamed: 13'],axis=1)

df2["date"] = pd.to_datetime(df2["date"])
df1['timestamp']=pd.to_datetime(df1['timestamp'])
df1['timestamp'] = df1["timestamp"].apply( lambda df1 : datetime.datetime(year=df1.year, month=df1.month, day=df1.day))
df2=df2.groupby('date').count()
df1=df1.groupby('timestamp').count()
df2['date']=df2.index
df1['timestamp']=df1.index

df1['Date'] = df1["timestamp"].apply( lambda df1 : datetime.datetime(year=df1.year, month=df1.month, day=df1.day))	
df1.set_index(df1["Date"],inplace=True)


df2['Date'] = df2["date"].apply( lambda df2 : datetime.datetime(year=df2.year, month=df2.month, day=df2.day))	
df2.set_index(df2["Date"],inplace=True)
print(df1['likes'])
df1=df1['text'].resample('W', how='sum')
df2=df2['text'].resample('W', how='sum')

fig = plt.figure()
plt.plot(df1,label="Tweets",marker='o')
plt.plot(df2,label="Reviews",marker='o')
plt.legend()
fig.suptitle('Google Data', fontsize=15)
plt.xlabel('Date', fontsize=15)
plt.ylabel('Total number', fontsize=15)
plt.show()
