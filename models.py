import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    
    def __init__(self, enc_hidden_dim, dec_hidden_dim, embedding_dim, 
                 latent_dim, vocab_size, dropout, seq_len):
        super().__init__()

        self.seq_len = seq_len

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, enc_hidden_dim)
        self.fc1 = nn.Linear(enc_hidden_dim, latent_dim)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(latent_dim, dec_hidden_dim)      
        self.decoder = nn.LSTM(dec_hidden_dim, dec_hidden_dim)
        self.fc3 = nn.Linear(dec_hidden_dim, vocab_size)
    
    def encode(self, x):
        # print(x.size())
        x = self.embedding(x).permute(1,0,2) # [T,B,E]=10*32*200
        _, (hidden, _) = self.encoder(x)
        z = self.fc1(hidden) # [1,B,L] #1*32*100
        z = self.dropout(z)
        return z
    
    def decode(self, z):
        z = self.fc2(z) # [1,B,H_dec] = 1*32*100 -> 1 *32 * 600
        out, _ = self.decoder(z.repeat(self.seq_len,1,1), (z, z))
        out = out.permute(1,0,2) # [B,T,H_dec]=32*10*600
        logits = self.fc3(out) # [B,T,V]=32*10*2703
        return logits.transpose(1,2) #=32*2703*10
    
    def forward(self, x):
        z = self.encode(x) 
        logits = self.decode(z)
        # print(logits.size())
        # print(z.squeeze().size(), logits.size())
        return (z.squeeze(), logits) #32*100, 32*2703*10


class Block(nn.Module):
    
    def __init__(self, block_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(block_dim, block_dim),
            nn.ReLU(True),
            nn.Linear(block_dim, block_dim),
        )
    
    def forward(self, x):
        return self.net(x) + x

class Generator(nn.Module):
    
    def __init__(self, n_layers, block_dim):
        super().__init__()

        self.net = nn.Sequential(
            *[Block(block_dim) for _ in range(n_layers)]
        )

    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    
    def __init__(self, n_layers, block_dim):
        super().__init__()

        self.net = nn.Sequential(
            *[Block(block_dim) for _ in range(n_layers)]
        )

    def forward(self, x):
        return self.net(x)
