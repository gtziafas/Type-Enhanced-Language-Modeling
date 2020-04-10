from torch.nn import Module, Sequential, ModuleList, Linear, LayerNorm, Embedding, Dropout, Tanh, KLDivLoss, Identity
from torch.nn import functional as F
from torch import LongTensor, Tensor
from torch.optim import Optimizer

import torch

from typing import Tuple, Optional, Callable, Sequence, TypeVar, Any, Dict, List, overload

tensor_map = Callable[[Tensor], Tensor]

ints = List[int]
Sample = Tuple[ints, ints]
Samples = List[Sample]
