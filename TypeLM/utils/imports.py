from torch.nn import Module, Sequential, ModuleList, Linear, LayerNorm, Embedding
from torch.nn import functional as F
from torch import LongTensor, Tensor

import torch

from typing import Tuple, Optional, Callable, Sequence, TypeVar, Any, Dict

tensor_map = Callable[[Tensor], Tensor]
tensor_maps = Sequence[tensor_map]
