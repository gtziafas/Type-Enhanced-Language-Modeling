from src.utils.imports import *


def masked_loss_wrapper(loss_fn):
    def masked_loss_fn(predictions, truth, masks):
        """
        :param predictions: FloatTensor of size S x D
        :param truth: LongTensor of size S( with low 0 and high D
        :param masks: BoolTensor of size S where 0 is excluded and 1 is included
        """
        predictions = predictions[masks == 1]
        truth = truth[masks == 1]
        return loss_fn(predictions, truth)
    return masked_loss_fn


# core_loss = torch.nn.CrossEntropyLoss(reduction='mean')
# masked_loss = masked_loss_wrapper(core_loss)


# todo. actual implementation