import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        """
        Depends on the Embed Size, i.e. the total embedding size coming in, and the number of heads that is to be distributed amongst.
        E.g. -> if embed_size = 256, and heads is 8, then each head will of dimension 32.
        """
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embed Size Non-Divisible By Heads, Please set so divisible."

        #Define linear layer that we send K,Q,V through

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads*self.head_dim, embed_size)


    def forward(self, values, keys, query, mask):

        #Number of samples we send at the same time.
        N = query.shape[0]

        #Within that samples, The respective lengths.
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        #Split embedding into self.head
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = keys.reshape(N, query_len, self.heads, self.head_dim)
        
        energy = torch.einsum("nqhd, nkhd --> nhqk", [queries, keys])
        #Queries shape: (N, query_len, heads, heads_dim)
        #Keys shape: (N, key_len, heads, heads_dim)
        #energy: (N, heads, query_len, key_len)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim = 3)
        out = torch.einsum("nhql,nlhd-->nqhd", [attention, values]).reshape(
            N, query_len, self.heads**self.head_dim
        )
        #Attention shape: (N, heads, query_len, key_len)
        #Values shape: (N, value_len, heads, heads_dim)
        # (N, query_len, heads, head_dim)
        #After Einsum, flatten last two dimensions.

        out = self.fc_out(out)
        return out
