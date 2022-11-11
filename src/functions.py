import torch
import torchtext
#from torchinfo import summary
import collections

all_amino=['A','C','D','E','F','G','H','K','I','L','M','N','P','Q','R','S','T','V','W','Y'] #all aminoacids in alphabetical order

def char_tokenizer(characters):
    return list(characters) 

def build_vocab(ngrams=1,min_freq=1):
    counter=collections.Counter()
    counter.update(torchtext.data.utils.ngrams_iterator(char_tokenizer(all_amino+['?']),ngrams=ngrams)) 
    vocab = torchtext.vocab.vocab(counter, min_freq=min_freq)
    return vocab
vocab=build_vocab()
    
def encode(x):
    return torch.LongTensor([vocab.get_stoi().get(s,0) for s in char_tokenizer(x)])

def load_dataset(inputfile, remove_n, max_input_data, max_eval_data, ntoks): 
    inpf=open(inputfile,'r')
    sequences=[]
    for line in inpf.readlines():
        if remove_n:
            sequences.append(line.replace('\n','').replace('n','').replace('X',''))
        else:
            sequences.append(line)
    half_Nwanted=len(sequences)

    all_data=[]
    ndata=0
    for seq in sequences:
        for i in range(len(seq)-ntoks): 
            shortened_seq = seq[i:i+ntoks]
            for n in range(ntoks): 
                shortened_seq_with_missing = shortened_seq # This is necessary not to keep track of previous aminoacids assigned as '?'
                shortened_seq_with_missing = shortened_seq_with_missing[0:n]+'?'+shortened_seq_with_missing[n+1:]
                if (ndata < (max_input_data + max_eval_data) ):
                    all_data.append((encode(shortened_seq_with_missing),encode(shortened_seq),shortened_seq))
                    ndata+=1
    print("Data size:",len(all_data))
    
    train_data=[]
    test_data=[]
    for i in range(len(all_data)):
        if (i<max_input_data):
            train_data.append(all_data[i])
        else:
            test_data.append(all_data[i])
            

    return train_data,test_data
