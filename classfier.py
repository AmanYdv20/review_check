'''This is the initial file for finding out the most relevent features for text document.
So this file will be run before active learning to find the best features'''

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from textblob import TextBlob
from pre_processing import preprocessing
import pickle
from scipy import sparse
from scipy.sparse import hstack
from sklearn.metrics import confusion_matrix
from check_stemer import tokenize
from finding_corpus import findCorpus
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score

Corpus=pd.read_csv('labelled_tweets.csv',encoding="latin-1")
Corpus=Corpus.drop(['Unnamed: 7'],axis=1)
Corpus=Corpus.dropna()
Corpus=Corpus[Corpus['Bug_report'].apply(lambda x: str(x).isdigit())]
Corpus.Bug_report = pd.to_numeric(Corpus.Bug_report, errors='coerce')
Corpus=Corpus.reset_index(drop=True)

'''code for finding the the sentiment score of text documents'''
sentiment_tweet=[]
for i in range(len(Corpus)):
    blob = TextBlob(Corpus['text'][i])
    ans=blob.sentiment
    sentiment_tweet.append(ans[0])
    
#X_senti = np.array(sentiment_tweet)
senti_train=sentiment_tweet[:719]
senti_test=sentiment_tweet[719:]
senti_train = np.array(senti_train).reshape(719,1)
senti_test = np.array(senti_test).reshape(309,1)
senti_train = sparse.csr_matrix(senti_train)
senti_test = sparse.csr_matrix(senti_test)
#X_senti = np.array(sentiment_tweet)

def convert(text):
    return str(text)

'''Classifier_final.csv is a file which contains all the preprocessed text if you have a unprocessed
text file then you can call preprocessing steps from here and can further proceed '''
Corpus=pd.read_csv('classifier_final.csv')

corpus_class=findCorpus(Corpus)

final_data=corpus_class.final_data

Corpus['text_final']=pd.Series(final_data)
Corpus['text_final']=Corpus['text_final'].apply(convert)
#pre=preprocessing(data)
#df=pre.data
 
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['text_final'],Corpus['Bug_report'],test_size=0.3)
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)

Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(Corpus['text_final'])

Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

#*******************************************
'''This section is for combining the Tf-idf vector with topic modelling result
Note that here I have given the size, you can change according to yours.'''

filename = 'assigned_topics.pickle'
infile = open(filename,'rb')
test_pic = pickle.load(infile)
infile.close()

train_pic=test_pic[:719]
test_pic=test_pic[719:]

X_train = np.array(train_pic)
X_train = sparse.csr_matrix(X_train) 
X_test = np.array(test_pic)
X_test = sparse.csr_matrix(X_test)
#*******************************************
final_train=hstack((Train_X_Tfidf, senti_train))
#combined_train=hstack((final_train,senti_train))
final_test=hstack((Test_X_Tfidf, senti_test))
#combined_test=hstack((final_test,senti_test))
#final_data=np.concatenate((Train_X_Tfidf, X_topic), axis=1)

'''Code for parameter tuning'''
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001, 0.00001, 10]}
 
# Make grid search classifier
clf_grid = GridSearchCV(svm.SVC(), param_grid, verbose=1)
 
# Train the classifier
clf_grid.fit(Train_X_Tfidf, Train_Y)
 
# clf = grid.best_estimator_()
print("Best Parameters:\n", clf_grid.best_params_)
print("Best Estimators:\n", clf_grid.best_estimator_)

#************************************

Naive = naive_bayes.MultinomialNB()
Naive.fit(final_train,Train_Y)

predictions_NB = Naive.predict(final_test)
print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)

#********************************************************************************
#SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')

SVM = svm.SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,decision_function_shape='ovr', degree=3, gamma=1, kernel='rbf', max_iter=-1,probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)
SVM.fit(Train_X_Tfidf,Train_Y)
predictions_SVM = SVM.predict(Test_X_Tfidf)
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)
confusion_matrix(Test_Y, predictions_SVM)
score=f1_score(predictions_SVM, Test_Y,average='binary')
print(score)

cross_val_score(svm.SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,decision_function_shape='ovr', degree=3, gamma=1, kernel='rbf', max_iter=-1,probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False),Train_X_Tfidf,Train_Y,cv=5)

#Corpus=Corpus.reset_index(drop=True)
unlabelled=pd.read_csv('unlabelled_data.csv',encoding="latin-1")
unlabelled=unlabelled.drop(['Unnamed: 0'],axis=1)
oracle = pd.read_csv('labelled_tweets.csv',encoding="latin-1")
oracle = oracle.drop(['Unnamed: 7'],axis=1)
oracle_filtered = oracle.dropna()
oracle_filtered_ids_list = oracle_filtered['tweet-id'].tolist()
mask = unlabelled['tweet-id'].isin(oracle_filtered_ids_list)
unlabel_df = unlabelled[~mask]
pre=preprocessing(unlabel_df)
unlabel_df=pre.data
unlabel_df['text']=unlabel_df['text'].apply(tokenize)
unlabel_df.to_csv("unlabel.csv", index=False)

oracle_filtered.to_csv('seed.csv',index=False)
