import torch
import torch.nn.functional as F

def generate_sample(
                model,
                vectorizer,
                sample_size=280,
                rough=True,
                capitalization=False,
                verbose=False):
    
    vocab = vectorizer.vocab
    model.to('cpu')
    model.eval()
    beginid = vocab.begin_idx
    begintensor = torch.tensor([beginid], dtype=torch.int64).unsqueeze(dim=0)
    ind = [begintensor]
    t = 1
    x_t = ind[1-1]
    h_t = None

    for t in range(1,sample_size+1):
        x_t = ind[t-1]
        emb_t = model.emb(x_t)
        rnn_t, h_t = model.rnn(emb_t, h_t)
        pred_vector = model.fc(rnn_t.squeeze(dim=1))
        # this squeezing is equivalent to the reshaping procedure in the model itself
        # this is due to batch_len = 1

        # ADD TODO softmax temperature!
        # Would be really cool to have something scale for how 'creative' the AI can be!
        prob_vector = F.softmax(pred_vector, dim=1)
        winner = torch.multinomial(prob_vector, num_samples=1)
        ind.append(winner)

    s = ""
    for i in range(len(ind)):
        idx = ind[i].item()
        s += vocab.lookup_idx(idx)
        
    if rough:
        if verbose:
            print("Generating rough word.")
        return s

    else:
        if verbose:
            print("Generating not rough word.")
        start = vocab.begin_token
        end = vocab.end_token
        out = s[s.find(start)+len(start):s.find(end)]
        
        if capitalization:
            out = out.capitalize()
        return out

def gen_samp(model,
    vocab,
    sample_size=120,
    prompt=""):

    bos = vocab.begin_idx

    one_hot = [bos]
    for c in prompt:
        idx = vocab.lookup_token(c)
        one_hot.append(idx)

    hot_tensor = torch.tensor(one_hot, dtype=torch.int64).unsqueeze(dim=0)
    embedded = model.emb(hot_tensor)
    _, h_n = model.rnn(embedded)
    # h_n contains the last outputs of all layers
    pred = model.fc(h_n[-1,:,:])
    prob = F.softmax(pred,dim=1)
    win = torch.multinomial(prob,num_samples=1)
    idx=win.item()
    one_hot.append(idx)

    for i in range(100):
        embedded = model.emb(win)
        _, h_n = model.rnn(embedded, h_n)
        pred = model.fc(h_n[-1,:,:])
        prob = F.softmax(pred, dim=1)
        win = torch.multinomial(prob,num_samples=1)
        one_hot.append(win.item())

    output = ""
    for idx in one_hot:
        token = vocab.lookup_idx(idx)
        output += token

    start = vocab.begin_token
    end = vocab.end_token
    return output[output.find(start)+len(start):output.find(end)]



