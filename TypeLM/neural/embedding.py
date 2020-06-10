from torch.nn import Module
import torch
from torch import LongTensor, Tensor
from torch.nn.functional import linear, embedding
from math import sqrt


class InvertibleEmbedder(Module):
    def __init__(self, embedding_dim: int, num_embeddings: int, padding_idx: int, scale_by_sqrt: bool):
        super(InvertibleEmbedder, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.embedding_matrix = torch.nn.Parameter(data=torch.rand(num_embeddings, embedding_dim), requires_grad=True)
        self.padding_idx = padding_idx
        self.embedding_scale = sqrt(embedding_dim) if scale_by_sqrt else 1.

    def embed(self, idxes: LongTensor) -> Tensor:
        emb = embedding(idxes, self.embedding_matrix, self.padding_idx) * self.embedding_scale
        return emb + positional_encoding(idxes.shape[0], idxes.shape[1], self.embedding_dim, device=str(emb.device))

    def invert(self, weights: LongTensor):
        return linear(weights, self.embedding_matrix.t())


def positional_encoding(b: int, n: int, d_model: int, freq: int = 10000, device: str = 'cpu') -> Tensor:
    pe = torch.zeros(n, d_model, device=device)
    position = torch.arange(0, n, device=device, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device, dtype=torch.float) *
                         - (torch.log(torch.tensor(freq, dtype=torch.float, device=device)) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.repeat(b, 1, 1)
