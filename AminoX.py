import sys
sys.path.append('./src')
from model import *
from plots import *
#
import numpy as np
import os
#
#from tensorboardX import SummaryWriter
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

def main():
    ntoks = args.min_length # Sequences will be of length ntoks. It must be taken less or equal to the minimum sequence length in the dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:",device)

    print("Output will be written in:",args.output_directory)
    if not os.path.exists(args.output_directory): os.system('mkdir '+args.output_directory)
    
    ### ### ### ### ### ### ### MAIN

    train_dataset,test_dataset = load_dataset(args.input_directory+args.input_name, args.remove_n, args.max_input_data, args.max_eval_data, ntoks)

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

if __name__ == '__main__':
    main()
