import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    embedding_dim: int = 4096 #The Embedding Dimension of the Model
    n_layers: int = 32
    n_heads: int = 32 
    num_kv_heads: Optional[int] = None 
    vocab_size: int = -1
    multiple_of: int = 256
    ffm_dim_multiplier: Optional[float]
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: int = 20248

def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    assert head_dim % 2 == 0, "Dimension must be divisible by 2"
    #Construct the theta score for each head dimension
    theta_numerator = torch.arange(0, head_dim, 2).float()
    theta = 1.0 / (theta ** (theta_numerator / head_dim))
    #The position of every value in the sequence len
    m = torch.arange(seq_len, device=device)


    freqs = torch.outer(m, theta).float()
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

class Transformer(nn.Module):
    
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        assert args.vocab_size != -1, "Vocab size must be set"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.token_embeddings = nn.Embedding(self.vocab_size, args.embedding_dim)
        
        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))

        self.norm = RMSNorm(args.embedding_dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        self.freqs_complex = precompute_theta_pos_frequencies(self.args.embedding_dim // self.args.n_heads, self.args.max_seq_len * 2, device=self.args.device)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        # (B, sq)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "Only one token at a time"

        # (B, sq) -> (B, sq, dim)
        h = self.token_embeddings(tokens)
        # Retrieve the pairs (m, theta) corresponding to the positions 

        freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]

        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h = self.norm(h)
        output = self.output(h).float()
        return output