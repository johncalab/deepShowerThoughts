"""
Assumes:
    vectorizer is a charVectorizer
    corpus is a pandas Series
        use: corpus = pd.read_csv('lastnames_clean.txt',header=None)[0]
"""
from torch.utils.data import Dataset
class charDataset(Dataset):
    def __init__(self,vectorizer,posts):
        self.posts = posts
        self.vectorizer = vectorizer

        max_len = len(posts.iloc[0])
        for sentence in posts:
            max_len = max(max_len, len(sentence))

        self.max_len = max_len + 3

    def __len__(self):
        return len(self.posts)
    
    def __getitem__(self,i):
        sent = self.posts.iloc[i]
        x,y = self.vectorizer.vectorize(sent=sent, max_len=self.max_len)
        assert x.shape == y.shape
        assert x.shape[0] == self.max_len-1
        return x,y
