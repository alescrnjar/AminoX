import numpy as np
import matplotlib.pyplot as plt

def plot_losses(Loss_mean, output_directory):
    fig = plt.figure(1, figsize=(4, 4))
    plt.plot(np.array(Loss_mean)[:, 0], np.array(Loss_mean)[:, 1],lw=1,c='C0')
    #plt.legend(loc='upper right',prop={'size':15})
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    fig.savefig(output_directory+'Loss.png',dpi=150)
    #plt.show()
    plt.clf()

def plot_prediction_matrix(prediction_matrix, all_amino, output_directory):
    fig = plt.figure(1, figsize=(4, 4))
    plt.imshow(prediction_matrix, interpolation='none')
    plt.xticks(np.arange(len(all_amino)), all_amino)
    plt.yticks(np.arange(len(all_amino)), all_amino)
    cbar = plt.colorbar(boundaries = np.linspace(min(np.array(prediction_matrix).reshape(-1)),max(np.array(prediction_matrix).reshape(-1)),10) , cmap='coolwarm') #.set_ticks(np.arange(10)))
    plt.xlabel('Predicted')
    plt.ylabel('Expected')
    cbar.set_label('Norm. prediction freq.') #,size=18)
    fig.savefig(output_directory+'Prediction_Matrix.png',dpi=150)
    #plt.show()
    plt.clf()
