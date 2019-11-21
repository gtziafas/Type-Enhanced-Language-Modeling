from torch.nn import Module, Sequential, ModuleList, Linear, LayerNorm
from torch.nn import functional as F
from torch import FloatTensor, LongTensor, Tensor

import torch

from typing import Tuple, Optional, Callable, Sequence, TypeVar, Any

tensor_map = Callable[[Tensor], Tensor]
tensor_maps = Sequence[tensor_map]
