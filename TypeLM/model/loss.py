from TypeLM.utils.imports import *
from TypeLM.utils.utils import sigsoftmax


def masked_loss_wrapper(loss_fn):
    def masked_loss_fn(predictions, truth, masks):
        predictions = predictions[masks == 1]
        truth = truth[masks == 1]
        return loss_fn(predictions, truth)
    return masked_loss_fn


class MixedLoss(Module):
    def __init__(self, language_loss: Module, type_loss: Module, language_loss_kwargs: Dict,
                 type_loss_kwargs: Dict, type_loss_weight: float) -> None:
        super(MixedLoss, self).__init__()
        self.language_loss = language_loss(**language_loss_kwargs)
        self.type_loss = type_loss(**type_loss_kwargs)
        self.kappa = type_loss_weight

    def forward(self, word_predictions: Tensor, word_truth: LongTensor,
                type_predictions: Tensor, type_truth: LongTensor, mask: LongTensor) -> Tuple[Tensor, Tensor]:
        return self.get_both_losses(word_predictions, word_truth, type_predictions, type_truth, mask)

    def get_both_losses(self, word_predictions: Tensor, word_truth: LongTensor,
                        type_predictions: Tensor, type_truth: LongTensor, mask: LongTensor) -> Tuple[Tensor, Tensor]:
        language_loss = masked_loss_wrapper(self.language_loss)(word_predictions, word_truth, mask)
        type_loss = self.type_loss(type_predictions, type_truth)
        return language_loss, type_loss


def label_smoothing(x: LongTensor, num_classes: int, smoothing: float, ignore_index: Optional[int] = None) -> Tensor:
    x_float = torch.ones(x.shape, device=x.device, dtype=torch.float).unsqueeze(-1)
    x_float = x_float.repeat([1 for _ in x.shape] + [num_classes])
    x_float.fill_(smoothing / (num_classes - 1))
    x_float.scatter_(dim=-1, index=x.unsqueeze(-1), value=1-smoothing)
    if ignore_index is not None:
        mask = x == ignore_index
        x_float[mask.unsqueeze(-1).repeat([1 for _ in x.shape] + [num_classes])] = 0
    return x_float


class LabelSmoother(Module):
    def __init__(self, num_classes: int, smoothing: float, ignore_index: Optional[int] = None):
        super(LabelSmoother, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def forward(self, x: LongTensor, ignore_index: Optional[int] = None,
                smoothing: Optional[float] = None) -> Tensor:
        return label_smoothing(x, self.num_classes,
                               smoothing if smoothing is not None else self.smoothing,
                               ignore_index if ignore_index is not None else self.ignore_index)


class FuzzyLoss(Module):
    def __init__(self, num_classes: int, mass_redistribution: float, ignore_index: int = 0,
                 reduction: str = 'batchmean') -> None:
        super(FuzzyLoss, self).__init__()
        self.loss_fn = KLDivLoss(reduction=reduction)
        self.label_smoother = LabelSmoother(num_classes, mass_redistribution, ignore_index)

    def __call__(self, x: Tensor, y: LongTensor) -> Tensor:
        smooth_y = self.label_smoother(y)
        return self.loss_fn(torch.log_softmax(x, dim=-1), smooth_y)


class CrossEntropySS(Module):
    def __init__(self, **kwargs):
        super(CrossEntropySS, self).__init__()
        self.NLL = torch.nn.NLLLoss(**kwargs)

    def forward(self, predictions: Tensor, truth: LongTensor) -> Tensor:
        predictions = sigsoftmax(predictions, dim=-1)
        return self.NLL(predictions.log(), truth)



# def example():
#     generic_loss = torch.nn.CrossEntropyLoss
#     generic_kwargs = {'ignore_index': 0, 'reduction': 'mean'}
#     mixed_loss = MixedLoss(generic_loss, generic_loss, generic_kwargs, generic_kwargs, 0.1)
#     wp = torch.rand(32, 12, 5, device='cuda').view(-1, 5)
#     tp = torch.rand(32, 12, 27, device='cuda').view(-1, 27)
#     wt = torch.randint(size=(32, 12), low=0, high=5, device='cuda').view(-1)
#     tt = torch.randint(size=(32, 12), low=0, high=27, device='cuda').view(-1)
#     wm = torch.randint(size=(32, 12), low=0, high=2, device='cuda').view(-1)
#     print(mixed_loss(wp, wt, tp, tt, wm))
