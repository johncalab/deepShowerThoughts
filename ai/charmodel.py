import torch.nn as nn
import torch.nn.functional as F

class charModel(nn.Module):
    def __init__(self,vocab_size,
                    embedding_dim=10,
                    rnn_hidden_dim=9,
                    padding_idx=0,
                    dropout_p=0.5,
                    num_layers=3,
                    bidirectional=False):
        super(charModel,self).__init__()

        self.dropout_p = dropout_p

        self.emb = nn.Embedding(num_embeddings=vocab_size,
                                embedding_dim=embedding_dim,
                                padding_idx=padding_idx)

        self.rnn = nn.GRU(input_size=embedding_dim,
                            hidden_size=rnn_hidden_dim,
                            dropout=dropout_p,
                            bidirectional=bidirectional,
                            num_layers=num_layers,
                            batch_first=True)
        if bidirectional:
            self.fc = nn.Linear(in_features=2*rnn_hidden_dim,
                out_features=vocab_size)
        else:
            self.fc = nn.Linear(in_features=rnn_hidden_dim,
                out_features=vocab_size)

    def forward(self, x_in, dropout=False, apply_softmax=False, verbose=False):
        
        if verbose:
            print(f"Input has shape {x_in.shape}.")

        x = self.emb(x_in)
        if verbose:
            print(f"Output of embedding layer has shape {x.shape}.")

        x,_ = self.rnn(x)
        if verbose:
            print(f"Output of RNN has shape {x.shape}.")
        
        batch_size, seq_size, _ = x.shape
        # contiguous: pytorch requires you to reallocate memory appropriately before reshaping
        x = x.contiguous().view(batch_size * seq_size, -1)
        if verbose:
            print(f"Reshaped output of RNN has shape {x.shape}.")

        x = self.fc(x)
        if verbose:
            print(f"Output of fc has shape {x.shape}.")

        if dropout:
            x = F.dropout(x,p=self.dropout_p)
        
        if apply_softmax:
            x = F.softmax(x,dim=1)
        
        x = x.view(batch_size, seq_size, -1)
        if verbose:
            print(f"Final output has shape {x.shape}.")
        return x