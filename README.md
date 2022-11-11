# AminoX

AminoX is a natural language processing (NLP) recurrent neural network (RNN) for the determination of a single missing aminoacid in a given input primary sequence:

<p align="center">
'SPSSLSTNTTSA ? PTLTSEPR'   &#8594  'SPSSLSTNTTSA S PTLTSEPR'
</p>

The input dataset is generated with ProtGPT2, a language model trained on protein space (https://huggingface.co/nferruz/ProtGPT2). Protgpt2_seq_gen.py allows for the generation of N different aminoacid sequences, of length comprised between a settable minimum (100) and a settable maximum (300).

The input data is the organised in minibatches of length 100, corresponding to the minimum length of the sequences. Each minibatch correspond to a sequence where, in turn, each aminoacid is substituted with the character '?' to represent a missing aminoacid, and its target output will be the unmodified sequence. 

AminoX is adapted from Unit 6/7 of this NLP tutorial: https://learn.microsoft.com/en-us/training/modules/intro-natural-language-processing-pytorch/1-introduction

# Required Libraries

Python modules required:

* numpy >= 1.22.3

* torch >= 1.12.1+cu116

* matplotlib.pyplot >= 3.4.3

# Example Prediction Matrix

<p align="center">
<img width="500" src=https://github.com/alescrnjar/AminoX/blob/main/example_output/Prediction_Matrix.png>
</p>