from nltk.util import ngrams
from collections import Counter

def finding_ngram(df,min_count,n=2):
    text=''
    for i in df['text']:
        text+=i+' '
    tokenized=text.split()
    esngrams = ngrams(tokenized, n)
    esngramFreq = Counter(esngrams)
    most_frequent=esngramFreq.most_common(min_count)
    print(most_frequent)
    f = open( str(n)+'grams.txt', 'w' )
    f.write( 'most frequent bigrams= ' + repr(most_frequent) + '\n' )
    f.close()
