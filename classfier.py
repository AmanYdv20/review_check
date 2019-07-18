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

Corpus=pd.read_csv('labelled_tweets.csv',encoding="latin-1")
Corpus=Corpus.drop(['Unnamed: 7'],axis=1)
Corpus=Corpus.dropna()
Corpus=Corpus[Corpus['Bug_report'].apply(lambda x: str(x).isdigit())]
Corpus.Bug_report = pd.to_numeric(Corpus.Bug_report, errors='coerce')
Corpus=Corpus.reset_index(drop=True)

#blob = TextBlob(Corpus['text'][100])
#ans=blob.sentiment

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

Corpus=pd.read_csv('classifier_final.csv')

corpus_class=findCorpus(Corpus)

final_data=corpus_class.final_data

Corpus['text_final']=pd.Series(final_data)
Corpus['text_final']=Corpus['text_final'].apply(convert)
#pre=preprocessing(data)
#df=pre.data
# Step - a : Remove blank rows if any.
#Corpus['text'].dropna(inplace=True)

# Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
#Corpus['text'] = [entry.lower() for entry in Corpus['text']]

#Corpus['text']=Corpus['text'].apply(remove_stopwords)
#Corpus['text']=Corpus['text'].apply(lemmatization)
# Step - c : Tokenization : In this each entry in the corpus will be broken into set of words
#Corpus['text']= [word_tokenize(entry) for entry in Corpus['text']]

# Step - d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
 
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['text_final'],Corpus['Bug_report'],test_size=0.3)
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)

Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(Corpus['text_final'])

Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)
#*******************************************
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

#Code for parameter tuning
from sklearn.model_selection import GridSearchCV
# Grid Search
# Parameter Grid
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

from sklearn.ensemble import RandomForestClassifier

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

import pandas as pd
import numpy as np
from sklearn import model_selection,linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import KFold
from sklearn import decomposition, ensemble

#from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from changeSlang import replaceAllSlang

def split_train_test(dataframe, train_feature_name, train_label):
    """splits a given dataframe to train and validation sets"""
    train_x, valid_x, train_y, valid_y = model_selection.train_test_split(
        dataframe[train_feature_name], dataframe[train_label])
    return train_x, valid_x, train_y, valid_y

def build_feature(text_data, max_features):
    """Builds a tfidf vector for the given data"""
    Tfidf_vect = TfidfVectorizer(max_features)
    Tfidf_vect.fit(unlabel['comment'])
    
    x_train_tfidf = Tfidf_vect.transform(text_data)
    return x_train_tfidf

def train_model_accuracy_calculator(train_feature_data, train_label, max_features):
    """Trains with the given data with SVM. Includes a split method for k-fold cross validation (k=7)"""
    
    model = svm.SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,decision_function_shape='ovr', degree=3, gamma=1, kernel='rbf', max_iter=-1,probability=True, random_state=None, shrinking=True, tol=0.001, verbose=False)
    scores=cross_val_score(model,build_feature(train_feature_data,max_features), train_label,cv=7)
    accuracy_r2_score = np.mean(scores)
    
    model=model.fit(build_feature(train_feature_data,max_features),Train_Y)
    return model, accuracy_r2_score

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
    
    oracle_filtered = oracle.dropna()
    
    oracle_filtered_ids_list = oracle_filtered['tweet-id'].tolist()

    mask = unlabelled['tweet-id'].isin(oracle_filtered_ids_list)
    unlabel_df = unlabelled[~mask]
    unlabel_df.to_csv("unlabel.csv", index=False)
    
    pd.concat([oracle_filtered, seed], ignore_index= True).to_csv("classifier_test.csv", index=False)

def convert(text):
    return str(text)

seed = pd.read_csv("classifier_test.csv")
#seed= seed.drop(['margins'],axis=1)
unlabel = pd.read_csv("test_data.csv")
unlabel['comment']=unlabel['comment'].apply(replaceAllSlang)

seed_class=findCorpus(seed)
final_data=seed_class.final_data
seed['comment']=pd.Series(final_data)
seed['comment']=seed['comment'].apply(convert)

unlabel_class=findCorpus(unlabel)
final_data=unlabel_class.final_data
unlabel['comment']=pd.Series(final_data)
unlabel['comment']=unlabel['comment'].apply(convert)


Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(seed['Bug_report'])

seed_comment_feature = build_feature(seed["comment"], max_features=15000)
unlabel_comment_feature = build_feature(unlabel["comment"],max_features= 15000)
print(unlabel_comment_feature.shape)
print(seed_comment_feature.shape)

model, accuracy = train_model_accuracy_calculator(train_feature_data=seed['comment'], train_label=Train_Y,max_features=5000)
print(model)
print(accuracy)

margins = uncertainty(make_predictions(classifier_model=model,predic_feature_vetor=unlabel_comment_feature))
query(margins, unlabel, "margins", 100)
#print(min(margins))

bug_report=model.predict(unlabel_comment_feature)

unlabel['bug_report']=bug_report
unlabel=unlabel[unlabel['text'].apply(lambda x: len(x.split(' ')) > 4)]

unlabel.to_csv('final_classification.csv',index=False)
update_files(query_file="query.csv", seed_file="classifier_test.csv", unlabel_file="unlabel.csv")
