"""
Assumes:
    vocab is a charVocabulary class
"""
import numpy as np
class charVectorizer(object):
    def __init__(self,vocab):
        self.vocab = vocab

    def vectorize(self, sent, max_len=-1):
        """
        max_len is used to know how much to pad
        """
        ind = [self.vocab.begin_idx]
        ind.extend(self.vocab.lookup_token(token) for token in sent)
        ind.append(self.vocab.end_idx)
        
        max_len = max(len(ind), max_len)

        x = np.empty(max_len-1, dtype=np.int64)
        x[:len(ind)-1] = ind[:-1]
        x[len(ind)-1:] = self.vocab.mask_idx

        y = np.empty(max_len-1, dtype=np.int64)
        y[:len(ind)-1] = ind[1:]
        y[len(ind)-1:] = self.vocab.mask_idx

        return x,y