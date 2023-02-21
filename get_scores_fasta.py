import sys
from transformers import AutoModel, AutoTokenizer, AutoModelForTokenClassification
import torch
import numpy as np
import torch.nn.functional as F
import pickle
from itertools import groupby
import pandas as pd

def fasta_iter(fasta_name):
    """
    modified from Brent Pedersen
    Correct Way To Parse A Fasta File In Python
    given a fasta file. yield tuples of header, sequence
    """
    fh = open(fasta_name)

    # ditch the boolean (x[0]) and just keep the header or sequence since
    # we know they alternate.
    faiter = (x[1] for x in groupby(fh, lambda line: line[0] == ">"))
    
    for header in faiter:
        # drop the ">"
        headerStr = header.__next__()[1:].strip()
    
        # join all sequence lines to one.
        seq = "".join(s.strip() for s in faiter.__next__())
        #print(seq)

        yield (headerStr, seq)

def get_out(sent):
    sent = sent[:min(1022, len(sent))]
    encoded = tokenizer.encode_plus(sent, return_tensors="pt")
    with torch.no_grad():
        output = model(**encoded)
    output = F.softmax(torch.squeeze(output['logits']))[1:-1,1].detach().numpy()
    #output = torch.squeeze(output['logits'])[1:-1,1].detach().numpy()
    return output
    

def get_all_outs(fasta_name):
    fiter = fasta_iter(fasta_name)
    ids = []
    seqs = []
    scores = []
    for ff in fiter:
        headerStr, sequence = ff
        ids.append(headerStr)
        seqs.append(sequence)
        scores.append(get_out(sequence))
    df = pd.DataFrame(
    {'ID': ids,
     'sequence': seqs,
     'score': scores
    })
    return df
    
checkpoint = sys.argv[1]
input_path = sys.argv[2]
output_path = sys.argv[3]

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForTokenClassification.from_pretrained(checkpoint)
model = model.eval()

out_df = get_all_outs(input_path)
out_df.to_pickle(output_path) 