"""
This script generates input dataset for AminoX.py
"""

import random
from transformers import pipeline

max_length= 300 
min_length= 100

wanted= 200 

num_return= 5 

outname = '../example_input/protgpt2_sequences.txt'
outf = open(outname,'w')

### ### ###

all_amino = ['A','C','D','E','F','G','H','K','I','L','M','N','P','Q','R','S','T','V','W','Y']

protgpt2 = pipeline('text-generation', model="nferruz/ProtGPT2")

accepted_seqs=[]

made=0
while made<wanted:
    print("made: {}/{}, {} unique sequences".format(made,wanted,len(accepted_seqs)))
    first_amino = random.choice(all_amino) 
    sequences = protgpt2(first_amino, max_length=max_length, do_sample=True, top_k=950, repetition_penalty=1.2, num_return_sequences=num_return, eos_token_id=0)
    
    for seq in sequences:
        myseq=seq['generated_text'].replace('\n','n')
        if myseq!=first_amino+'n' and (len(myseq)-myseq.count('n'))>=min_length and (len(myseq)-myseq.count('n'))<=max_length and myseq not in accepted_seqs:
            accepted_seqs.append(myseq)
            outf.write(myseq+'\n')
            made+=1
            if (made==wanted): break

outf.close()
print("DONE: "+outname)
