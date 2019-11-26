from src.utils.imports import *
from torch.nn.utils.rnn import pad_sequence


def masked_loss_wrapper(loss_fn):
    def masked_loss_fn(predictions, truth, masks):
        predictions = predictions[masks == 1]
        truth = truth[masks == 1]
        return loss_fn(predictions, truth)
    return masked_loss_fn


# core_loss = torch.nn.CrossEntropyLoss(reduction='mean')
# masked_loss = masked_loss_wrapper(core_loss)

def convert_indices_to_bool_masks(masks: Sequence[LongTensor]) -> LongTensor:
    return pad_sequence(list(masks), batch_first=True, padding_value=0)
