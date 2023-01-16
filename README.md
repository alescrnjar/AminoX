# AminoX

AminoX is a Natural Language Processing (NLP) Recurrent Neural Network (RNN) for the determination of a single missing aminoacid in a given input primary sequence:

<p align="center">
'SPSSLSTNTTSA ? PTLTSEPR'   &#8594  'SPSSLSTNTTSA S PTLTSEPR'
</p>

The input dataset is generated with ProtGPT2, a language model trained on protein space (https://huggingface.co/nferruz/ProtGPT2). Protgpt2_seq_gen.py allows for the generation of N different aminoacid sequences, of length comprised between a settable minimum (100) and a settable maximum (300).

The input data is then organised in minibatches of length 100, corresponding to the minimum length of the sequences. Each minibatch correspond to a sequence where, in turn, each aminoacid is substituted with the character '?' to represent a missing aminoacid, and its target output will be the unmodified sequence. 

After training epochs, predictions are made over the test set. For each amino acid in this dataset, the whole list of aminoacids is shown, in decreasing order of prediction likelihood. A confusion matrix is also plotted, in order to show how aminoacids are correctly/incorrectly predicted.

AminoX is adapted from Unit 6/7 of this NLP tutorial: https://learn.microsoft.com/en-us/training/modules/intro-natural-language-processing-pytorch/1-introduction

# Required Libraries

Python modules required:

* numpy >= 1.22.3

* torch >= 1.12.1+cu116 (pip install torch==1.12.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html)

* torchtext >= 0.13.1

* matplotlib.pyplot >= 3.4.3

* transformers >= 4.22.2 (required by Protgpt2_seq_gen.py)

# Example Confusion Matrix

<p align="center">
<img width="500" src=https://github.com/alescrnjar/AminoX/blob/main/example_output/Prediction_Matrix.png>
</p>