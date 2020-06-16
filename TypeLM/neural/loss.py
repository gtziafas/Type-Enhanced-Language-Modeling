from torch.nn import Module, KLDivLoss
from typing import Dict, Tuple, List
from torch import Tensor, LongTensor
import torch


def masked_loss_wrapper(fn):
    def wrapped(predictions: Tensor, truth: LongTensor, mask: LongTensor):
        predictions = predictions[mask == 1]
        truth = truth[mask == 1]
        return fn(predictions, truth)
    return wrapped


class MixedLoss(Module):
    def __init__(self, lang_loss: Module, type_loss: Module, lang_loss_kwargs: Dict,
                 type_loss_kwargs: Dict, type_loss_scale: float = 1.0):
        super(MixedLoss, self).__init__()
        self.lang_loss = lang_loss(**lang_loss_kwargs)
        self.type_loss = type_loss(**type_loss_kwargs)
        self.type_loss_scale = type_loss_scale

    def forward(self, word_predictions: Tensor, word_truth: LongTensor, type_predictions: Tensor,
                type_truth: LongTensor, mask: LongTensor) -> Tuple[Tensor, Tensor]:
        lang_loss, type_loss = self.get_both_losses(word_predictions, word_truth, type_predictions, type_truth, mask)
        return lang_loss, type_loss * self.type_loss_scale

    def get_both_losses(self, word_predictions: Tensor, word_truth: LongTensor, type_predictions: Tensor,
                        type_truth: LongTensor, mask: LongTensor) -> Tuple[Tensor, Tensor]:
        lang_loss = masked_loss_wrapper(self.lang_loss)(word_predictions, word_truth, mask)
        type_loss = self.type_loss(type_predictions, type_truth)
        return lang_loss, type_loss


class FuzzyLoss(Module):
    def __init__(self, reduction: str, num_classes: int,
                 mass_redistribution: float, ignore_index: List[int]) -> None:
        super(FuzzyLoss, self).__init__()
        self.loss_fn = KLDivLoss(reduction=reduction)
        self.nc = num_classes
        self.mass_redistribution = mass_redistribution
        self.ignore_index = ignore_index

    def __call__(self, x: Tensor, y: LongTensor) -> Tensor:
        y_float = torch.zeros_like(x, device=x.device, dtype=torch.float)
        y_float.fill_(self.mass_redistribution / (self.nc-(1 + len(self.ignore_index))))
        y_float.scatter_(1, y.unsqueeze(1), 1 - self.mass_redistribution)
        mask = torch.zeros_like(y, dtype=torch.bool)
        for idx in self.ignore_index:
            mask = mask.masked_fill_(y.bool(), 1) 
        y_float[mask.unsqueeze(1).repeat(1, self.nc)] = 0
        return self.loss_fn(torch.log_softmax(x.view(-1, self.nc), dim=-1), y_float)
