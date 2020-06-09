from torch.nn import Module
import torch
from torch import LongTensor, Tensor
from torch.nn.functional import linear, embedding
from math import sqrt


class InvertibleEmbedder(Module):
    def __init__(self, embedding_dim: int, num_embeddings: int, padding_idx: int, scale_by_sqrt: bool):
        super(InvertibleEmbedder, self).__init__()
        self.embedding_matrix = torch.nn.Parameter(data=torch.rand(num_embeddings, embedding_dim), requires_grad=True)
        self.padding_idx = padding_idx
        self.embedding_scale = sqrt(embedding_dim) if scale_by_sqrt else 1.

    def embed(self, idxes: LongTensor) -> Tensor:
        return embedding(idxes, self.embedding_matrix, self.padding_idx) * self.embedding_scale

    def invert(self, weights: LongTensor):
        return linear(weights, self.embedding_matrix.t())
