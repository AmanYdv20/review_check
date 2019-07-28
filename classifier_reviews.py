import pandas as pd
import numpy as np
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from textblob import TextBlob
from pre_processing import preprocessing
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag
import pickle
from scipy import sparse
from scipy.sparse import hstack
from sklearn.metrics import confusion_matrix
from check_stemer import tokenize
#from finding_corpus import remove_stopwords, lemmatization
from finding_corpus import findCorpus
from sklearn.model_selection import cross_val_score
#import topic_modeling

def feature_tfidf(data):
    return Tfidf_vect.transform(data)

def feature_tfidf_topic(data):    
    data_tfidf=feature_tfidf(data)
    data_topics=topic_modeling.find_topics(data)
    data_topics = np.array(data_topics)
    data_topics = sparse.csr_matrix(data_topics)
    final_train=hstack((data_tfidf, data_topics))
    return final_train


    
    

filename = 'assigned_topics.pickle'
infile = open(filename,'rb')
test_pic = pickle.load(infile)
infile.close()

Corpus=pd.read_csv('seed.csv')
Corpus=Corpus.drop(['Unnamed: 0'],axis=1)
Corpus=Corpus.dropna()
Corpus=Corpus[Corpus['Bug_report'].apply(lambda x: str(x).isdigit())]
Corpus.Bug_report = pd.to_numeric(Corpus.Bug_report, errors='coerce')
Corpus=Corpus.reset_index(drop=True)


sentiment_tweet=[]
for i in range(len(Corpus)):
    blob = TextBlob(Corpus['text'][i])
    ans=blob.sentiment
    sentiment_tweet.append(ans[0])
    
senti_train = np.array(sentiment_tweet).reshape(1497,1)
senti_train = sparse.csr_matrix(senti_train)


def convert(text):
    return str(text)

corpus_class=findCorpus(Corpus)

final_data=corpus_class.final_data

Corpus['text_final']=pd.Series(final_data)
Corpus['text_final']=Corpus['text_final'].apply(convert)
#Corpus.to_csv('seed.csv')

Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(Corpus['text_final'])

test_mat=Tfidf_vect.transform(Corpus['text_final'])

data_topics = np.array(test_pic)
data_topics = sparse.csr_matrix(data_topics)
final_train=hstack((test_mat, data_topics))

final_train=hstack((final_train, senti_train))

Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(final_train,Corpus['Bug_report'],test_size=0.3)
Test_X, Val_X, Test_Y, Val_Y = model_selection.train_test_split(Test_X,Test_Y,test_size=0.35)

#Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(test_mat,Corpus['Bug_report'],test_size=0.3)
#Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['text_final'],Corpus['Bug_report'],test_size=0.3)
#Test_X, Val_X, Test_Y, Val_Y = model_selection.train_test_split(Test_X,Test_Y,test_size=0.35)


Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)
Val_Y = Encoder.fit_transform(Val_Y)


Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)
Val_X_Tfidf = Tfidf_vect.transform(Val_X)


data_Tfidf = Tfidf_vect.transform(Corpus['text_final'])
data_Y = Encoder.fit_transform(Corpus['Bug_report'])

#Code for parameter tuning
from sklearn.model_selection import GridSearchCV
# Grid Search
# Parameter Grid
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001, 0.00001, 10]}
 
# Make grid search classifier
clf_grid = GridSearchCV(svm.SVC(), param_grid, verbose=1)
 
# Train the classifier
clf_grid.fit(data_Tfidf, data_Y)

print("Best Estimators:\n", clf_grid.best_estimator_)


SVM = svm.SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,decision_function_shape='ovr', degree=3, gamma=1, kernel='rbf', max_iter=-1,probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)
SVM.fit(Train_X,Train_Y)
predictions_SVM = SVM.predict(Test_X)
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)

confusion_matrix(predictions_SVM,Test_Y)

val_SVM = SVM.predict(Val_X)
print("SVM Accuracy Score -> ",accuracy_score(val_SVM, Val_Y)*100)
confusion_matrix(val_SVM,Val_Y)


filename = 'assigned_topics.pickle'
infile = open(filename,'rb')
test_pic = pickle.load(infile)
infile.close()

print(test_pic[0])

model = svm.SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,decision_function_shape='ovr', degree=3, gamma=1, kernel='rbf', max_iter=-1,probability=True, random_state=None, shrinking=True, tol=0.001, verbose=False)
scores=cross_val_score(model,final_train,Encoder.fit_transform(Corpus['Bug_report']) ,cv=10)
accuracy_r2_score = np.mean(scores)
