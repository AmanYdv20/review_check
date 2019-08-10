import pandas as pd
import numpy as np
from sklearn import model_selection,linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import KFold
from sklearn import decomposition, ensemble
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import cross_val_score

#from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from changeSlang import replaceAllSlang
from finding_corpus import findCorpus
from sklearn.preprocessing import LabelEncoder
from textblob import TextBlob
from scipy import sparse
from scipy.sparse import hstack

lb = LabelBinarizer()

def split_train_test(dataframe, train_feature_name, train_label):
    """splits a given dataframe to train and validation sets"""
    train_x, valid_x, train_y, valid_y = model_selection.train_test_split(
        dataframe[train_feature_name], dataframe[train_label])
    return train_x, valid_x, train_y, valid_y

def build_feature(text_data, senti_text, text_score, max_features):
    """Builds a tfidf vector for the given data"""
    sentiment_tweet=[]
    for i in range(len(senti_text)):
        blob = TextBlob(str(senti_text[i]))
        ans=blob.sentiment
        sentiment_tweet.append(ans[0])
    senti_train = np.array(sentiment_tweet).reshape(len(senti_text),1)
    senti_train = sparse.csr_matrix(senti_train)
    Tfidf_vect = TfidfVectorizer(max_features)
    Tfidf_vect.fit(data['comment'])
    x_train_tfidf = Tfidf_vect.transform(text_data)
    print(x_train_tfidf.shape)
    final_train=hstack((x_train_tfidf, senti_train))
    print(final_train.shape)
    text_score=list(text_score)
    text_score_mat = np.array(text_score).reshape(len(text_score),1)
    text_score_mat = sparse.csr_matrix(text_score_mat.astype(float))
    train_data = hstack((final_train, text_score_mat))
    print(train_data.shape)
    return train_data
    #return final_train

'''def build_feature(text_data, max_features):
    """Builds a tfidf vector for the given data"""
    Tfidf_vect = TfidfVectorizer(max_features)
    Tfidf_vect.fit(unlabel['comment'])
    
    x_train_tfidf = Tfidf_vect.transform(text_data)
    
    
    return x_train_tfidf
'''

def train_model_accuracy_calculator(train_feature_data, train_label, max_features):
    """Trains with the given data with SVM. Includes a split method for k-fold cross validation (k=7)"""
    
    model = svm.SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,decision_function_shape='ovr', degree=3, gamma=1, kernel='rbf', max_iter=-1,probability=True, random_state=None, shrinking=True, tol=0.001, verbose=False)
    scores=cross_val_score(model,train_feature_data, train_label,cv=10)
    accuracy_r2_score = np.mean(scores)
    y_train = np.array([number[0] for number in lb.fit_transform(train_label)])
    recall = cross_val_score(model,train_feature_data, y_train,cv=10, scoring='recall')
    final_recall=np.mean(recall)
    precision = cross_val_score(model,train_feature_data, y_train,cv=10, scoring='precision')
    final_precision=np.mean(precision)
    f1_score = cross_val_score(model,train_feature_data, y_train,cv=10, scoring='f1')
    f1_score=np.mean(f1_score)
    model=model.fit(train_feature_data,Train_Y)
    print(scores)
    return model, accuracy_r2_score, final_recall,final_precision,f1_score

def make_predictions(classifier_model, predic_feature_vetor):
    """make predictions for the unseen data"""
    results = classifier_model.predict_proba(predic_feature_vetor)
    return results

def uncertainty(prediction_results):
    """uncertainty function to select the next query"""
    margins = []
    for i in range(0, len(prediction_results)):
        margins.append(abs(prediction_results[i,0] - prediction_results[i,1]))
    return margins

def query(margins, dataframe, margines_column, k):
    """chooses the top k selected data and write to query.csv file"""
    dataframe[margines_column] = margins
    dataframe.sort_values(by = [margines_column], ascending=True, inplace=True)
    #un_labelled_data.to_csv("send_for_oracle.csv")
    dataframe.nsmallest(n=k, columns=[margines_column]).to_csv("query.csv", index = False)
    
def update_files(query_file, unlabel_file, seed_file):
    """reads the labeled query file and adds the labeled data to seed file, removes the labeled data from unlabel data"""
    unlabelled = pd.read_csv(unlabel_file)
    seed = pd.read_csv(seed_file)    
    oracle = pd.read_csv(query_file)
    oracle=oracle[500:]
    oracle=oracle[oracle['comment'].apply(lambda x: len(str(x).split(',')) > 1)]
    oracle_filtered = oracle.dropna()
    #oracle_filtered = oracle
    oracle_filtered_ids_list = oracle_filtered.id.tolist()

    mask = unlabelled['id'].isin(oracle_filtered_ids_list)
    unlabel_df = unlabelled[~mask]
    unlabel_df.to_csv("unlabel.csv", index=False)
    
    pd.concat([oracle_filtered, seed], ignore_index= True).to_csv("seed.csv", index=False)

def convert(text):
    return str(text)

seed = pd.read_csv("seed.csv")
seed=seed.drop(['Unnamed: 0','index','text_final','margins','original_text'],axis=1)
seed_class=findCorpus(seed)
final_data=seed_class.final_data
seed['comment']=pd.Series(final_data)
seed['comment']=seed['comment'].apply(convert)

seed=seed.dropna()
seed=seed.reset_index()
#seed= seed.drop(['margins'],axis=1)
unlabel = pd.read_csv("unlabel.csv")
#unlabel.dropna()
#unlabel=unlabel[unlabel['Bug_report'].apply(lambda x: str(x).isdigit())]
#unlabel['comment']=unlabel['comment'].apply(replaceAllSlang)
#unlabel=unlabel.drop(['Unnamed: 9','Unnamed: 10','Unnamed: 11','Unnamed: 12','Unnamed: 13','Unnamed: 14','Unnamed: 15'],axis=1)
#unlabel.to_csv('unlabel.csv')

'''
seed = pd.read_csv("classifier_test.csv")
#seed= seed.drop(['margins'],axis=1)
unlabel = pd.read_csv("test_data.csv")
unlabel['comment']=unlabel['comment'].apply(replaceAllSlang)
'''


#unlabel_class=findCorpus(unlabel)
#final_data=unlabel_class.final_data
#unlabel['comment']=pd.Series(final_data)
unlabel['comment']=unlabel['comment'].apply(convert)

df=['0','1','2','3','4','5']
b=[0,1,2,3,4,5]
unlabel = unlabel[unlabel['score'].isin(df) | unlabel['score'].isin(b)]

unlabel=unlabel.reset_index()

unlabel['score']=pd.to_numeric(unlabel['score'])
seed['score']=pd.to_numeric(seed['score'])

#unlabel = unlabel[(unlabel[['score']] != '/').all(axis=1)]

#unlabel.to_csv('unlabel.csv',index=False)

Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(seed['Bug_report'])

seed_comment_feature = build_feature(seed["comment"], seed['text'], seed['score'],max_features=15000)
unlabel_comment_feature = build_feature(unlabel["comment"],unlabel['text'], unlabel['score'],max_features= 15000)
print(unlabel_comment_feature.shape)
print(seed_comment_feature.shape)

#train_feature=build_feature(seed['comment'],seed['text'],seed['score'],max_features=15000)

model, accuracy,recall_score, precision_score,f1_score = train_model_accuracy_calculator(train_feature_data=seed_comment_feature, train_label=Train_Y,max_features=15000)
print(model)
print("Accuracy ",accuracy)
print("Recall ",recall_score)
print("Precision score ", precision_score)
print("f1-score ", f1_score)
#confusion matrix

results=make_predictions(classifier_model=model,predic_feature_vetor=unlabel_comment_feature)
margins = uncertainty(results)
query(margins, unlabel, "margins", 500)
#print(min(margins))

#bug_report=model.predict(unlabel_comment_feature)

#unlabel['bug_report']=bug_report
#unlabel=unlabel[unlabel['text'].apply(lambda x: len(x.split(' ')) > 4)]

#unlabel.to_csv('final_classification.csv',index=False)
update_files(query_file="query.csv", seed_file="seed.csv", unlabel_file="unlabel.csv")

'''oracle = pd.read_csv('query.csv')
oracle=oracle[:500]
oracle=oracle[oracle['comment'].apply(lambda x: len(str(x).split(',')) > 1)]

type(oracle['comment'][0])
print(len(oracle['comment'][0]))'''

data=pd.read_csv('unlabel_final.csv')
data=data[data['comment'].apply(lambda x: len(str(x).split(' ')) > 2)]
data['comment']=data['comment'].apply(convert)
df=['0','1','2','3','4','5']
b=[0,1,2,3,4,5]
data = data[data['score'].isin(df) | data['score'].isin(b)]

data=data.reset_index()
data_comment_feature = build_feature(data["comment"],data['text'], data['score'],max_features= 15000)
predictions_SVM = model.predict(data_comment_feature)
data['Bug_report']=predictions_SVM
data.to_csv('final_review_file',index=False)

data=pd.read_csv('final_review_file.csv')
data1.groupby(['appTitle']).agg(['count'])
data1=data[data['Bug_report']==1]
data1.to_csv('bug_report_data_reviews.csv',index=False)
data2=data[data['Bug_report']==0]