from TypeLM.utils.imports import *
from TypeLM.utils.utils import type_accuracy
from TypeLM.model.loss import MixedLoss
from TypeLM.model.type_factored_lm import TypeFactoredLM
from TypeLM.data.loader import LazyLoader


from torch.optim import Optimizer

from torch.nn.utils.rnn import pad_sequence as _pad_sequence


two_ints = Tuple[int, int]


def pad_sequence(x):
    return _pad_sequence(x, batch_first=True, padding_value=0)


def train_batch(model: TypeFactoredLM, masked_words: LongTensor, true_words: LongTensor, types: LongTensor,
                pad: LongTensor, masked_indices: LongTensor, loss_fn: MixedLoss,
                optimizer: Optimizer) -> Tuple[float, two_ints, two_ints]:
    model.train()

    # total num of samples
    num_samples = masked_words.shape[0] * masked_words.shape[1]

    # forward pass and loss
    word_preds, type_preds = model(masked_words, pad)
    sent_stats, t_stats = type_accuracy(type_preds.argmax(dim=-1), types, 0)
    batch_loss = loss_fn(word_preds.view(num_samples, -1), true_words.flatten(),
                         type_preds.view(num_samples, -1), types.flatten(), masked_indices.view(-1))

    # backprop
    batch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return batch_loss.item(), sent_stats, t_stats


def train_batches(model: TypeFactoredLM, dl: LazyLoader, loss_fn: MixedLoss, optimizer: Optimizer, num_batches: int,
                  device: str) -> Tuple[float, float, float]:
    batch_idx, epoch_loss = 0, 0.
    sum_sent, sum_cor_sent, sum_words, sum_cor_words = 0, 0, 0, 0

    for batch_idx in range(num_batches):
        batch = dl.get_processed_batch()
        words_input, words_truth, types, pad, words_mask = list(map(lambda x: x.to(device), batch))
        batch_stats = train_batch(model, words_input, words_truth, types, pad, words_mask, loss_fn, optimizer)
        batch_loss, (num_sent, num_cor_sent), (num_words, num_cor_words) = batch_stats
        epoch_loss += batch_loss
        sum_sent += num_sent
        sum_cor_sent += num_cor_sent
        sum_words += num_words
        sum_cor_words += num_cor_words
    epoch_loss /= (batch_idx+1)

    return epoch_loss, sum_cor_sent/sum_sent, sum_cor_words/sum_words


def train_batch_st(model: TypeFactoredLM, masked_words: LongTensor, types: LongTensor,
                   pad: LongTensor, loss_fn: Module, optimizer: Optimizer) -> Tuple[float, two_ints, two_ints]:
    model.train()

    # total num of samples
    num_samples = masked_words.shape[0] * masked_words.shape[1]

    # forward pass and loss
    type_preds = model.forward_st(masked_words, pad)
    sent_stats, t_stats = type_accuracy(type_preds.argmax(dim=-1), types, 0)
    batch_loss = loss_fn(type_preds.view(num_samples, -1), types.flatten())

    # backprop
    batch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return batch_loss.item(), sent_stats, t_stats


def train_batches_st(model: TypeFactoredLM, dl: LazyLoader, loss_fn: MixedLoss, optimizer: Optimizer, num_batches: int,
                     device: str) -> Tuple[float, float, float]:

    batch_idx, epoch_loss = 0, 0.
    sum_sent, sum_cor_sent, sum_words, sum_cor_words = 0, 0, 0, 0

    for batch_idx in range(num_batches):
        batch = dl.get_processed_batch()
        words_input, _, types, pad, _ = list(map(lambda x: x.to(device), batch))
        batch_stats = train_batch_st(model, words_input, types, pad, loss_fn, optimizer)
        batch_loss, (num_sent, num_cor_sent), (num_words, num_cor_words) = batch_stats
        epoch_loss += batch_loss
        sum_sent += num_sent
        sum_cor_sent += num_cor_sent
        sum_words += num_words
        sum_cor_words += num_cor_words
    epoch_loss /= (batch_idx+1)

    return epoch_loss, sum_cor_sent/sum_sent, sum_cor_words/sum_words

