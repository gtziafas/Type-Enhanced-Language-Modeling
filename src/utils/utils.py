from src.utils.imports import *
from random import random


def positional_encoding(b: int, n: int, d_model: int, freq: int = 10000, device: str= 'cpu',
                        dropout_rate: float=0.1) -> FloatTensor:
    pe = torch.zeros(n, d_model, device=device)
    position = torch.arange(0, n, device=device, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device, dtype=torch.float) *
                         - (torch.log(torch.tensor(freq, dtype=torch.float, device=device)) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.repeat(b, 1, 1)

    return F.dropout(pe, dropout_rate)


def mask_indices(sen_len: int, random_chance: float) -> Sequence[int]:
    return [x for x in range(sen_len) if random() < random_chance]


_T1 = TypeVar('_T1')


def mask_sampling(sentence: Sequence[_T1], masked_indices: ints,
                  sampling_strategy: Callable[[Any], _T1], *args, **kwargs) -> Sequence[_T1]:
    return [t if i not in masked_indices else sampling_strategy(*args, **kwargs) for i, t in enumerate(sentence)]
