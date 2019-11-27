from TypeLM.utils.imports import *
from torch.nn.utils.rnn import pad_sequence


def masked_loss_wrapper(loss_fn):
    def masked_loss_fn(predictions, truth, masks):
        predictions = predictions[masks == 1]
        truth = truth[masks == 1]
        return loss_fn(predictions, truth)
    return masked_loss_fn


class MaskedLossWrapper(Module):
    def __init__(self, loss: Module):
        super(MaskedLossWrapper, self).__init__()
        self.loss = loss

    def forward(self, predictions, truth, masks):
        # KAPOTE THA SKASEI
        predictions = predictions[masks == 1]
        truth = truth[masks == 1]
        return self.loss(predictions, truth)


# core_loss = torch.nn.CrossEntropyLoss(reduction='mean')
# masked_loss = masked_loss_wrapper(core_loss)

def convert_indices_to_bool_masks(masks: Sequence[LongTensor]) -> LongTensor:
    return pad_sequence(list(masks), batch_first=True, padding_value=0)


class MixedLoss(Module):
    def __init__(self, language_loss: Module, type_loss: Module, language_loss_kwargs: Dict,
                 type_loss_kwargs: Dict, language_loss_weight: float) -> None:
        super(MixedLoss, self).__init__()
        self.language_loss = language_loss(**language_loss_kwargs)
        self.type_loss = type_loss(**type_loss_kwargs)
        self.kappa = language_loss_weight

    def forward(self, predictions: Tensor, truth: LongTensor, mask: LongTensor) -> Tensor:
        language_loss = masked_loss_wrapper(self.language_loss)(predictions, truth, mask)
        type_loss = masked_loss_wrapper(self.language_loss)(predictions, truth, mask)
        return language_loss + self.kappa * type_loss


# generic_loss = torch.nn.CrossEntropyLoss
# generic_kwargs = {'ignore_index': 0, 'reduction': 'none'}
# mixed_loss = MixedLoss(generic_loss, generic_loss, generic_kwargs, generic_kwargs, 0.1)
# p = torch.rand(32, 12, 5).view(-1, 5)
# t = torch.randint(size=(32, 12), low=0, high=5).view(-1)
# m = torch.randint(size=(32, 12), low=0, high=2).view(-1)
# print(masked_loss_wrapper(generic_loss(**generic_kwargs))(p, t, m))
# print(mixed_loss(p, t, m))


