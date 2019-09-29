"""
This is the main script to train a model.
It requires a 'source', from which it locates different files needed for training.

If -scratch: train model from scratch, using the corresponding csv file.
    In this case the script will save the trained model, and a pickled dictionary for decoding.
If -resume: continue training model, using corresponding model, dictionary, and csv file.
    In this case the script will only save the trained model.
"""
from charvocabulary import charVocabulary
from charvectorizer import charVectorizer
from chardataset import charDataset
from charmodel import charModel
from charsample import generate_sample
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import tqdm
import pickle
import traceback
import os
import argparse

parser = argparse.ArgumentParser()
# source (csv file, used to train from scratch), eg: maynov2017
parser.add_argument('-source', type=str)
# resume (if you want to resume training a model)
parser.add_argument('-resume', action='store_true', default=False)
# if -resume, we need a model (which model you want to resume training for)
# eg: maynov2017_169 [this will determine which dict and which model to look for]
parser.add_argument('-model', type=str)
# dict (if you want to specify a certain dictionary)
# eg: maynov2017_169_dict.pkl (the full name)
parser.add_argument('-dict', type=str)


# if -twitter it will post to twitter
parser.add_argument('-twitter', action='store_true', default=False)

# ne = number of epochs
parser.add_argument('-ne', type=int, default=88)
# dropout speeds up training by turning off neurons
parser.add_argument('-dropout', type=float, default=None)
# torch device
parser.add_argument('-cuda', action='store_true', default=False)
parser.add_argument('-device', type=str, default='cuda')

args = parser.parse_args()

# root paths
csv_path = os.path.join('source', args.source + '.csv')
dict_path = os.path.join('source', args.source + '_dict.pkl')
if args.dict:
    dict_path = os.path.join('source', args.dict)

if args.resume:
    model_path = os.path.join('source', args.model + '_model.pt')
else:
    from random import randint
    identifier = randint(100,999)
    identifier = str(identifier)
    model_path = os.path.join('source', args.source + '_' + identifier + '_model.pt')

print("OK, let's get this started.")
print(f"My training source is {csv_path}.")
print(f"I hope to train for {args.ne} epochs.")
if args.twitter:
    print("I will be posting on twitter.")
print(f"The model is located at {model_path}.")
print(f"The dictionary is located at {dict_path}.")

# Load csv ----------------------------------------------------------
posts = pd.read_csv(csv_path, index_col=[0]).title.astype('U')
print(f"Loaded. There are {len(posts)} posts. Let's move on to loading (or building) the vocabulary.")

# Create/Load vocabulary --------------------------------------------
if args.resume:
    token_to_idx = pickle.load(open(dict_path,'rb'))
    vocab = charVocabulary(token_to_idx=token_to_idx)

else:
    vocab = charVocabulary()
    vocab.add_series(df=posts)
    print("Vocabulary done. Let's save the bijection between tokens and ids, for later use.")
    pickle.dump(vocab.token_to_idx, open(dict_path,'wb'))

# maskid
maskid = vocab.mask_idx

# Load vectorizer
vectorizer = charVectorizer(vocab=vocab)

# create/load model
if args.resume:
    model = charModel(vocab_size=len(vocab),padding_idx=maskid)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
else:
    if args.dropout:
        model = charModel(vocab_size=len(vocab),padding_idx=maskid,dropout_p=args.dropout)
    else:
        model = charModel(vocab_size=len(vocab),padding_idx=maskid)

# Load dataset and dataloader ---------------------------------------
ds = charDataset(vectorizer=vectorizer, posts=posts)
dl = DataLoader(ds, batch_size=4, shuffle=True)

# optimizer
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# Twitter ----------------------------------------------------------------------------
from tweetthis import WHEN_TO_TWEET
if args.twitter:
    from tweetthis import tweetify
    from creds.twitter_cred import *
    import tweepy
    auth = tweepy.OAuthHandler(API_KEY,API_SECRET) 
    auth.set_access_token(ACCESS_TOKEN,ACCESS_SECRET)

    api = tweepy.API(auth, wait_on_rate_limit=True,wait_on_rate_limit_notify=True)

    try:
        api.verify_credentials()
        print("Twitter authentication OK")
    except:
        print("Error during authentication")

# set device ---------------------------------
if args.cuda and torch.cuda.is_available():
    device = torch.device(args.device)
else:
    device = torch.device('cpu')

# transfer model to device -----------------
model.to(device)
print(f"\nDevice used is {device}.")

# OK let's start training ----------------------------------------------------------
num_epochs = args.ne
try:
    for epoch in range(num_epochs):
        print(f"\nEpoch number {epoch+1} is starting now.")
        model.train()

        with tqdm.tqdm(total=len(dl)) as progress_bar:
            for i,data in enumerate(dl):
                x,y = data
                
                optimizer.zero_grad()

                x = x.to(device)
                y = y.to(device)

                y_pred = model(x)

                batch_size, seq_len, feats = y_pred.shape
                y_pred_loss = y_pred.view(batch_size*seq_len,feats)
                y_loss = y.view(-1)

                loss = F.cross_entropy(y_pred_loss, y_loss, ignore_index=maskid)
                loss.backward()
                optimizer.step()

                progress_bar.update(1)

                if args.twitter and i in WHEN_TO_TWEET:
                    j=1
                    while j <= 10:
                        # TODO: use tracebacks to see what error you get
                        # instead of trying to generate random sentences, try cutting the sentence in half?
                        try:
                            showt = generate_sample(model=model,vectorizer=vectorizer,rough=False)
                            tweet = tweetify(showt)
                            print("I will try to tweet:", tweet)
                            api.update_status(tweet)
                        except:
                            print("Tweeting failed.")
                            print(traceback.format_exc())
                            j+=1
                        else:
                            j = 11

        model.eval()

        model.to('cpu')
        for i in range(5):
            print(generate_sample(model=model,vectorizer=vectorizer))
        model.to(device)


        print(f"Epoch number {epoch+1} has now concluded. I am saving the model now.")
        torch.save(model.state_dict(), model_path)

except:
    print("\nTraining was interrupted. That's ok, I'll still save the latest model.")

torch.save(model.state_dict(), model_path)