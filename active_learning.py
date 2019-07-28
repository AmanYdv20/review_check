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

lb = LabelBinarizer()

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
    y_train = np.array([number[0] for number in lb.fit_transform(train_label)])
    recall = cross_val_score(model,build_feature(train_feature_data,max_features), y_train,cv=7, scoring='recall')
    final_recall=np.mean(recall)
    precision = cross_val_score(model,build_feature(train_feature_data,max_features), y_train,cv=7, scoring='precision')
    final_precision=np.mean(precision)
    f1_score = cross_val_score(model,build_feature(train_feature_data,max_features), y_train,cv=7, scoring='f1')
    f1_score=np.mean(f1_score)
    model=model.fit(build_feature(train_feature_data,max_features),Train_Y)
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
    
    oracle_filtered = oracle.dropna()
    
    oracle_filtered_ids_list = oracle_filtered['id'].tolist()

    mask = unlabelled['id'].isin(oracle_filtered_ids_list)
    unlabel_df = unlabelled[~mask]
    unlabel_df.to_csv("unlabel.csv", index=False)
    
    pd.concat([oracle_filtered, seed], ignore_index= True).to_csv("seed.csv", index=False)

def convert(text):
    return str(text)

seed = pd.read_csv("seed.csv")
#seed= seed.drop(['margins'],axis=1)
unlabel = pd.read_csv("test_data.csv")
unlabel['comment']=unlabel['comment'].apply(replaceAllSlang)

'''
seed = pd.read_csv("classifier_test.csv")
#seed= seed.drop(['margins'],axis=1)
unlabel = pd.read_csv("test_data.csv")
unlabel['comment']=unlabel['comment'].apply(replaceAllSlang)
'''
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

model, accuracy,recall_score, precision_score,f1_score = train_model_accuracy_calculator(train_feature_data=seed['comment'], train_label=Train_Y,max_features=5000)
print(model)
print(accuracy)
print(recall_score)
print(precision_score)
print(f1_score)
#confusion matrix

results=make_predictions(classifier_model=model,predic_feature_vetor=unlabel_comment_feature)
margins = uncertainty(results)
query(margins, unlabel, "margins", 100)
#print(min(margins))

bug_report=model.predict(unlabel_comment_feature)

unlabel['bug_report']=bug_report
unlabel=unlabel[unlabel['text'].apply(lambda x: len(x.split(' ')) > 4)]

unlabel.to_csv('final_classification.csv',index=False)
update_files(query_file="query.csv", seed_file="classifier_test.csv", unlabel_file="unlabel.csv")