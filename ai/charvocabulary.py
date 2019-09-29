class charVocabulary(object):
    def __init__(self, token_to_idx=None):
        if token_to_idx is None:
            token_to_idx = {}
        self.token_to_idx = token_to_idx
        self.idx_to_token = {idx: token 
                                for token, idx in self.token_to_idx.items()}

        self.mask_token = '<mask>'
        self.begin_token = '<begin>'
        self.end_token = '<end>'
        self.unk_token = '<unk>'
        self.space_token = ' '

        self.mask_idx = self.add_token(self.mask_token)
        self.begin_idx = self.add_token(self.begin_token)
        self.end_idx = self.add_token(self.end_token)
        self.unk_idx = self.add_token(self.unk_token)
        self.space_idx = self.add_token(self.space_token)

    def add_token(self, token):
        if token in self.token_to_idx:
            index = self.token_to_idx[token]
        else:
            index = len(self.token_to_idx)
            self.token_to_idx[token] = index
            self.idx_to_token[index] = token
        return index

    def __len__(self):
        assert len(self.token_to_idx) == len(self.idx_to_token)
        return len(self.token_to_idx)

    def lookup_token(self,token):
        return self.token_to_idx[token]

    def lookup_idx(self,i):
        return self.idx_to_token[i]

    def add_txt(self,path):
        with open(path, 'r') as f:
            fulltext = f.read()
            for c in fulltext:
                if c != '\n':
                    self.add_token(c)
        return None

    def add_series(self,df):
        for sentence in df:
            max_len = min(300, len(sentence))
            for char in sentence[:max_len]:
                self.add_token(char)
        return None