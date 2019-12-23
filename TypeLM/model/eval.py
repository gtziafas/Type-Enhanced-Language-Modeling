from TypeLM.utils.imports import *
from TypeLM.utils.utils import type_accuracy
from TypeLM.model.loss import MixedLoss
from TypeLM.model.type_factored_lm import TypeFactoredLM
from TypeLM.data.loader import EagerLoader


from torch.nn.utils.rnn import pad_sequence as _pad_sequence


two_ints = Tuple[int, int]


def pad_sequence(x):
    return _pad_sequence(x, batch_first=True, padding_value=0)


def eval_batch(model: TypeFactoredLM, masked_words: LongTensor, true_words: LongTensor, types: LongTensor,
                pad: LongTensor, masked_indices: LongTensor, loss_fn: MixedLoss) -> Tuple[float, two_ints, two_ints]:
    model.eval()

    # total num of samples
    num_samples = masked_words.shape[0] * masked_words.shape[1]

    # forward pass and loss
    word_preds, type_preds = model(masked_words, pad)
    sent_stats, t_stats = type_accuracy(type_preds.argmax(dim=-1), types, 0)
    batch_loss = loss_fn.type_loss(type_preds.view(num_samples, -1), types.flatten())

    return batch_loss.item(), sent_stats, t_stats


def eval_batches(model: TypeFactoredLM, dl: EagerLoader, loss_fn: MixedLoss, device: str) -> Tuple[float, float, float]:
    batch_idx, epoch_loss = 0, 0.
    sum_sent, sum_cor_sent, sum_words, sum_cor_words = 0, 0, 0, 0

    dl.make_line_iterator()

    while True:
        try:
            batch = dl.get_processed_batch()
        except StopIteration:
            break
        words_input, words_truth, types, pad, words_mask = list(map(lambda x: x.to(device), batch))
        batch_stats = eval_batch(model, words_input, words_truth, types, pad, words_mask, loss_fn)
        batch_loss, (num_sent, num_cor_sent), (num_words, num_cor_words) = batch_stats
        epoch_loss += batch_loss
        sum_sent += num_sent
        sum_cor_sent += num_cor_sent
        sum_words += num_words
        sum_cor_words += num_cor_words
        batch_idx += 1
    epoch_loss /= (batch_idx+1)

    return epoch_loss, sum_cor_sent/sum_sent, sum_cor_words/sum_words
