"""
Args
    -module: type of model (for now only 'char' is supported)
    -source: name of the model (expects folder with that name in ./ai)
    -prompt: optional prompt to start off a sentence
"""
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-module', type=str, default='char')
parser.add_argument('-source', type=str, required=True)
parser.add_argument('-prompt', type=str, default='')
args = parser.parse_args()

import os
root_path = os.path.join('ai', args.source)

import pickle
params_path = os.path.join(root_path, 'params.pkl')
params = pickle.load(open(params_path,'rb'))

model_path = os.path.join(root_path, 'model.pt')

if args.module == 'char':
    from ai.charmodel import charModel
    from ai.charvocabulary import charVocabulary
    from ai.charsample import gen_samp
    import torch
    
    dict_path = os.path.join(root_path, 'dict.pkl')
    token_to_idx = pickle.load(open(dict_path,'rb'))
    vocab = charVocabulary(token_to_idx=token_to_idx)

    model = charModel(**params)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    out = gen_samp(model=model, vocab=vocab, prompt=args.prompt)
    print(out)
else:
    print('Not yet')
    raise ValueError("Don't know that one.")
