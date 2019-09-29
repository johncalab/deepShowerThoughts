import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-prompt', type=str, default="")
# later add model and params?
# parser.add_argument('-model', type=str, required=True)
# parser.add_argument('-length', type=int, default=40)
# parser.add_argument('-rough', type=bool, default=False)
# parser.add_argument('-capitalization', type=bool, default=False)
args = parser.parse_args()

from charvocabulary import charVocabulary
from charmodel import charModel
from charsample import gen_samp
import torch
import pickle
import os

root_path = 'bob'
model_path = os.path.join(root_path, 'model.pt')
dict_path = os.path.join(root_path, 'dict.pkl')

token_to_idx = pickle.load(open(dict_path,'rb'))
vocab = charVocabulary(token_to_idx=token_to_idx)

params = {'vocab_size':len(vocab),
'embedding_dim': 128,
'rnn_hidden_dim': 512,
'num_layers': 2,}


model = charModel(**params)
# model = charModel(vocab_size=len(vocab),
#                       embedding_dim=EMBEDDING_DIM,
#                       rnn_hidden_dim=RNN_HIDDEN_DIM,
#                       padding_idx=maskid,
#                       dropout_p=DROPOUT,
#                       num_layers=NUM_LAYERS,
#                       bidirectional=BIDIRECTIONAL)

model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

out = gen_samp(model=model, vocab=vocab, prompt=args.prompt)
print(out)
