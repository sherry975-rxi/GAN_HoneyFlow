import collections

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re
import pandas as pd

class Vocab:

    def __init__(self, corpus):
        self.words = self._build(corpus)
        self.encoding = {w:i for i,w in enumerate(self.words)}
        self.decoding = {i:w for i,w in enumerate(self.words)}
        # print(self.words, ' words')
        # print(len(self.encoding), ' encoding')
        # print(len(self.decoding), ' decoding')

        self.register('<pad>')
        self.register('<unk>')
        self.register('<s>')
        self.register('</s>')
    
    def _build(self, corpus, clip=1):
        # vocab = collections.Counter()
        vocab = collections.defaultdict(int)
        # corpus = ['ultimately feels like just one more in the long line of films this year about', 'uncommonly pleasurable']
        for sent in corpus:
            for tokens in sent:
                if tokens == '':
                    continue
                vocab[tokens] = vocab.get(tokens, 0) + 1


        for word in list(vocab.keys()):
            if vocab[word] < clip:
                vocab.pop(word)
        
        return list(sorted(vocab.keys()))

    def register(self, token, index=-1):
        i = len(self.encoding) if index<0 else index
        self.encoding[token] = i
        self.decoding[i] = token

    def size(self):
        assert len(self.encoding) == len(self.decoding)
        return len(self.encoding)

class Corpus(Dataset):
    
    def __init__(self, seq_len=10):
        self.seq_len = seq_len
        self.sample_N = 50000
        self.reviews = self._load()
        self.vocab = Vocab(self.reviews)

    
    def _load(self):

        dataset = pd.read_csv('CIDDS-001-internal-week1-client1.csv', usecols=['Datefirstseen', 'Duration', 'SrcIPAddr', 'SrcPt', 'DstIPAddr', 'DstPt', 'Packets', 'Bytes'])

        packets = []

        for index, rows in dataset.head(self.sample_N).iterrows():
            row = [str(r) for r in rows]
        
            t_list = [str(rows.Datefirstseen), str(rows.Duration).replace(' ', ''), str(rows.SrcIPAddr).replace(' ', ''), 
            str(rows.SrcPt).replace(' ', ''), str(rows.DstIPAddr).replace(' ', ''), str(rows.DstPt).replace(' ', ''), str(rows.Packets).replace(' ', '')
                        ,str(rows.Bytes).replace(' ', '')]
            packets.append(t_list)

        # print(packets)
        return [x for x in packets]
    
    def pad(self, sample):
        l,r = 0,self.seq_len-len(sample)
        # print(len(sample))
        if r <= 0:
            return sample[:self.seq_len+1]
        else:
            return np.pad(sample, (0,r), 'constant')

    def encode(self, sample):
        enc = self.vocab.encoding
        unk_idx = enc['<unk>']
        return np.array([enc['<s>']]+[enc.get(c, unk_idx) \
                         for c in sample]+[enc['</s>']])
    
    def decode(self, sample):
        dec = self.vocab.decoding
        return ' '.join(np.array([dec[c.item()] for c in sample]))
    
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, i):
        return torch.from_numpy(self.pad(self.encode(self.reviews[i])))

def load(batch_size, seq_len):
    ds = Corpus(seq_len)
    return (DataLoader(ds, batch_size, shuffle=True), ds.vocab)



