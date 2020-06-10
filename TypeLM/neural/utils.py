from typing import overload, List
from torch.nn.utils.rnn import pad_sequence as _pad_sequence
from torch import Tensor, LongTensor


@overload
def pad_sequence(x: List[LongTensor], padding_value: int) -> LongTensor:
    pass


@overload
def pad_sequence(x: List[Tensor], padding_value) -> Tensor:
    pass


def pad_sequence(x, padding_value):
    return _pad_sequence(x, batch_first=True, padding_value=padding_value)