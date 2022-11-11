import torch
import torchtext
from torchinfo import summary
import sys
sys.path.append('./src')
from functions import *
from plots import *
#
import numpy as np
import random
import os
#
import collections
#
from tensorboardX import SummaryWriter
#
import argparse

parser = argparse.ArgumentParser()

# Input settings
parser.add_argument('--input_directory', default='./example_input/', type=str) 
parser.add_argument('--input_name', default='protgpt2_sequences.dat', type=str)
parser.add_argument('--min_length', default=100, type=int)
parser.add_argument('--max_input_data', default=50000, type=int) 
parser.add_argument('--max_eval_data', default=1000, type=int)  
parser.add_argument('--remove_n', default=True, type=bool)

# Training settings
parser.add_argument('--n_epochs', default=20, type=int) 
parser.add_argument('--batch_size', default=100, type=int) 
parser.add_argument('--hidden_dim', default=64, type=int) 
parser.add_argument('--learning_rate', default=0.001, type=float) 

# Output settings
parser.add_argument('--log_freq', default=1, type=int)
parser.add_argument('--output_directory', default='./example_output/', type=str) 

args = parser.parse_args()

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

ntoks = args.min_length # Sequences will be of length ntoks. It must be taken less or equal to the minimum sequence length in the dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:",device)

print("Output will be written in:",args.output_directory)
if not os.path.exists(args.output_directory): os.system('mkdir '+args.output_directory)

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

def load_dataset(): 
    inpf=open(args.input_directory+args.input_name,'r')
    sequences=[]
    for line in inpf.readlines():
        if args.remove_n:
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
                if (ndata < (args.max_input_data + args.max_eval_data) ):
                    all_data.append((encode(shortened_seq_with_missing),encode(shortened_seq),shortened_seq))
                    ndata+=1
    print("Data size:",len(all_data))
    
    train_data=[]
    test_data=[]
    for i in range(len(all_data)):
        if (i<args.max_input_data):
            train_data.append(all_data[i])
        else:
            test_data.append(all_data[i])
            

    return train_data,test_data

class LSTM_Filler(torch.nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.rnn1 = torch.nn.LSTM(vocab_size, hidden_dim, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, vocab_size)
    def forward(self, x, s=None):
        x = torch.nn.functional.one_hot(x,vocab_size).to(torch.float32)
        x,s = self.rnn1(x,s)
        return self.fc(x),s 

def evaluate(net, example):
    
    n = random.randint(0,len(example)-1)
    text_1 = example
    text_1 = text_1[0:n]+'?'+text_1[n+1:]
    test_out,s = net(encode(text_1).to(device))
    index = torch.argmax(test_out[n]) 
    amino = vocab.get_itos()[index]
    expected = example[n] 
    
    test_out_line_sorted, indexes = torch.sort(test_out[n],descending=True)
    descending_aminos = []
    for index in indexes:
        descending_aminos.append(vocab.get_itos()[index])
    return amino , expected , descending_aminos
    
### ### ### ### ### ### ### MAIN


train_dataset,test_dataset = load_dataset()

data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
print("Number of batches:",len(data_loader))

print("{} training sequences, {} test sequences".format(len(train_dataset),len(test_dataset)))

vocab_size = len(vocab)

print("Vocabulary:", vocab.get_stoi())

net = LSTM_Filler(vocab_size, args.hidden_dim).to(device)

optimizer = torch.optim.Adam(net.parameters(),lr=args.learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()
net.train() 
Loss_average = []
for epoch_idx in range(args.n_epochs):
    losses = []
    for batch_idx, data_input in enumerate(data_loader):
        text_in = data_input[0].to(device) 
        text_out = data_input[1].to(device) 
        optimizer.zero_grad()
        out,s = net(text_in)
        loss = torch.nn.functional.cross_entropy(out.view(-1,vocab_size),text_out.flatten()) 
        losses.append(loss.item())
        if (epoch_idx==0  and batch_idx==0): 
            print("Initial: loss: {}".format(loss.item()))
        loss.backward()
        optimizer.step()
    l_mean = torch.mean(torch.FloatTensor(losses))
    Loss_average.append([epoch_idx,l_mean])
    if epoch_idx % args.log_freq == 0:
        print("{}/{}: loss = {}".format(epoch_idx,args.n_epochs,l_mean)) 

plot_losses(Loss_average, args.output_directory)
        
print("Evaluation:")
prediction_matrix = []
for a1 in all_amino:
    prediction_matrix.append([])
    for a2 in all_amino:
        prediction_matrix[-1].append(0)
n_predictions_per_amino = {}
for a2 in all_amino:
    n_predictions_per_amino[a2] = 0

net.eval() #TODO 
count = 0
rate = 0.
for test in test_dataset:
    example=test[2]
    for t in range(3):
        count += 1
        predicted_amino, expected_amino , descending_predictions = evaluate(net, example)
        print("Predicted: {} Expected: {} ({})".format(predicted_amino, expected_amino, descending_predictions))
        rate += (predicted_amino == expected_amino)
        prediction_matrix[all_amino.index(predicted_amino)][all_amino.index(expected_amino)] += 1
        n_predictions_per_amino[expected_amino] += 1

for a1 in all_amino:
    for a2 in all_amino:
        if n_predictions_per_amino[a2] != 0 : prediction_matrix[all_amino.index(a1)][all_amino.index(a2)] /= n_predictions_per_amino[a2]
plot_prediction_matrix(prediction_matrix, all_amino, args.output_directory)

print("Success prediction rate:",rate/count)

