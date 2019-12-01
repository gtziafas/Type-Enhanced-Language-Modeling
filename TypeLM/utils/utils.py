from TypeLM.utils.imports import *
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


def count_parameters(model: Module) -> int:
    total_param = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_param = torch.prod(torch.tensor(param.size()))
            if param.dim() > 1:
                print(name, ':', 'x'.join(str(x) for x in list(param.size())), '=', num_param)
            else:
                print(name, ':', num_param)
            total_param += num_param
    return total_param


def mask_sample(sample: Sample, random_chance: float = 0.15, mask_token: str = '[MASK]') -> Tuple[Sample, ints]:
    words, types = list(zip(*sample))
    masked_indices = mask_indices(len(words), random_chance)
    # todo. actual sampling
    words = mask_sampling(words, masked_indices, lambda: mask_token)
    types = mask_sampling(types, masked_indices, lambda: mask_token)
    return list(zip(words, types)), masked_indices


def mask_indices(sen_len: int, random_chance: float) -> Sequence[int]:
    return [0 if random() < random_chance else 1 for _ in range(sen_len)]


_T1 = TypeVar('_T1')


def mask_sampling(sentence: Sequence[_T1], masked_indices: ints,
                  sampling_strategy: Callable[[Any], _T1], *args, **kwargs) -> Sequence[_T1]:
    return [t if masked_indices[i] == 0 else sampling_strategy(*args, **kwargs)
            for i, t in enumerate(sentence[:-1])]


# sentence = ['i', 'like', 'donkeys', 'and', 'chickens']
# mask = mask_indices(len(sentence), 0.5)
# mask
# [1, 0, 0, 0, 1]
# mask_sampling(sentence, mask, lambda : 'MASK')
# ['MASK', 'like', 'donkeys', 'and', 'MASK']