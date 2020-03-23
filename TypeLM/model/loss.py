from TypeLM.utils.imports import *


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
