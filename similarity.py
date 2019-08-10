import gensim
from gensim.models import Word2Vec
w2v_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary = True)
#It takes a while to load
print("model loaded")
