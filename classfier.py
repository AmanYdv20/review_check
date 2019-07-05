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

stoplist = stopwords.words('english')
my_stopwords = "whatsit atuser url" # my extra stopwords
stoplist = stoplist + my_stopwords.split()

Corpus=pd.read_csv('random_data_7000.csv')
Corpus=Corpus.drop(['Unnamed: 7','Unnamed: 8','Unnamed: 9'],axis=1)
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

Corpus=pd.read_csv('classifier_final.csv')
#pre=preprocessing(data)
#df=pre.data
# Step - a : Remove blank rows if any.
Corpus['text'].dropna(inplace=True)

# Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
Corpus['text'] = [entry.lower() for entry in Corpus['text']]

# Step - c : Tokenization : In this each entry in the corpus will be broken into set of words
Corpus['text']= [word_tokenize(entry) for entry in Corpus['text']]

# Step - d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.

# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

for index,entry in enumerate(Corpus['text']):
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stoplist and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    # The final processed set of words for each iteration will be stored in 'text_final'
    Corpus.loc[index,'text_final'] = str(Final_words)

 
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
#Train_X_Tfidf=np.asanyarray(Train_X_Tfidf)
final_train=hstack((Train_X_Tfidf, senti_train))
#combined_train=hstack((final_train,senti_train))
final_test=hstack((Test_X_Tfidf, senti_test))
#combined_test=hstack((final_test,senti_test))
#final_data=np.concatenate((Train_X_Tfidf, X_topic), axis=1)

Naive = naive_bayes.MultinomialNB()
Naive.fit(final_train,Train_Y)

predictions_NB = Naive.predict(final_test)
print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)

#********************************************************************************
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)
predictions_SVM = SVM.predict(Test_X_Tfidf)
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)
#Corpus=Corpus.reset_index(drop=True)

