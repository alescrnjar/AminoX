from functions import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTM_Filler(torch.nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.rnn1 = torch.nn.LSTM(vocab_size, hidden_dim, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, vocab_size)
    def forward(self, x, s=None):
        x = torch.nn.functional.one_hot(x,self.vocab_size).to(torch.float32)
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