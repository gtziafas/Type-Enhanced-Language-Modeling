from src.utils.imports import *
from random import random


def positional_encoding(b: int, n: int, d_model: int, freq: int = 10000, device: str = 'cpu',
                        dropout_rate: float = 0.1) -> Tensor:
    pe = torch.zeros(n, d_model, device=device)
    position = torch.arange(0, n, device=device, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device, dtype=torch.float) *
                         - (torch.log(torch.tensor(freq, dtype=torch.float, device=device)) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.repeat(b, 1, 1)

    return F.dropout(pe, dropout_rate)


def mask_indices(sen_len: int, random_chance: float) -> Sequence[int]:
    return [0 if random() < random_chance else 1 for _ in range(sen_len)]


_T1 = TypeVar('_T1')


def mask_sampling(sentence: Sequence[_T1], masked_indices: ints,
                  sampling_strategy: Callable[[Any], _T1], *args, **kwargs) -> Sequence[_T1]:
    return [t if masked_indices[i] == 0 else sampling_strategy(*args, **kwargs)
            for i, t in enumerate(sentence)]


# sentence = ['i', 'like', 'donkeys', 'and', 'chickens']
# mask = mask_indices(len(sentence), 0.5)
# mask
# [1, 0, 0, 0, 1]
# mask_sampling(sentence, mask, lambda : 'MASK')
# ['MASK', 'like', 'donkeys', 'and', 'MASK']
