import torch
import torch.nn as nn
import numpy as np
from pandas import DataFrame
from models import Autoencoder, Generator
from dataset import Corpus
from pandas.core.common import flatten

ds = Corpus()
vocab = ds.vocab
# print(vocab.encoding, ' is vocab' )

generator = Generator(20, 100)
generator.eval()
generator.load_state_dict(torch.load('generator.th', map_location='cpu'))

autoencoder = Autoencoder(100, 600, 200, 100, vocab.size(), 0.5, 10)
autoencoder.eval()
autoencoder.load_state_dict(torch.load('autoencoder.th', map_location='cpu'))

w_v = autoencoder.state_dict()['embedding.weight']
print('embedding size is', w_v.size())

df = DataFrame(columns = ['Datefirstseen', 'Duration', 'SrcIPAddr', 'SrcPt', 'DstIPAddr', 'DstPt', 'Packets', 'Bytes'])
while len(df) < (50000):
    # print(len(df))
    # sample noise
    noise = torch.from_numpy(np.random.normal(0, 1, (1,100))).float()
    z = generator(noise[None,:,:])
    # print('z size is', z.size())
    # create new sentence
    logits = autoencoder.decode(z).squeeze()
    # print('logits size is', logits.size())

    seq = logits.argmax(dim=0)
    pkt = ds.decode(seq)
      
    res = pkt.split(' ')
    # print(res)
    if '-' in res[1] and ':' in res[2]:
        time = res[1]+' '+res[2]
        temp = res[3:-1]
        res = temp.insert(0, time)
        # print(temp)
        try:
            df.loc[len(df)] = temp
        except:
            continue
    



df.to_csv('test_res_192_168_200_8.csv', index=False)