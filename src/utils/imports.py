from torch.nn import Module, Sequential, ModuleList, Linear
from torch.nn import functional as F
from torch import FloatTensor, LongTensor, Tensor

import torch

from typing import Tuple, Optional, Callable, Iterable

tensor_map = Callable[[Tensor], Tensor]
tensor_maps = Iterable[tensor_map]
