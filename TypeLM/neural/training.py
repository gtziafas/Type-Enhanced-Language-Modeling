from torch import LongTensor
from typing import Tuple
from TypeLM.neural.model import TypedLM
from TypeLM.neural.loss import MixedLoss
from TypeLM.neural.utils import pad_sequence
from TypeLM.preprocessing.loader import LazyLoader
from torch.optim import Optimizer
import torch


def type_accuracy(predictions: LongTensor, truth: LongTensor, ignore_idx: int) \
        -> Tuple[Tuple[int, int], Tuple[int, int]]:
    correct_items = torch.ones_like(predictions)
    correct_items[predictions != truth] = 0
    correct_items[truth == ignore_idx] = 1

    correct_sents = correct_items.prod(dim=1)
    num_correct_sents = correct_sents.sum().item()

    num_correct_items = correct_items.sum().item()
    num_masked_items = len(truth[truth == ignore_idx])

    return ((predictions.shape[0], num_correct_sents),
            (predictions.shape[0] * predictions.shape[1] - num_masked_items, num_correct_items - num_masked_items))


def train_batch(model: TypedLM, loss_fn: MixedLoss, optim: Optimizer, masked_words: LongTensor,
                padding_mask: LongTensor, true_words: LongTensor, true_types: LongTensor, masked_ids: LongTensor):
    model.train()

    num_words = masked_words.shape[0] * masked_words.shape[1]
    num_types = true_types.shape[0] * true_types.shape[1]

    type_pad_idx = model.tokenizer.type_tokenizer.PAD_TOKEN_ID

    word_preds, type_preds = model.forward_train(masked_words, padding_mask, true_types)
    sent_stats, type_stats = type_accuracy(type_preds.argmax(dim=-1), true_types, type_pad_idx)

    batch_losses = loss_fn(word_preds.view(num_words, -1), true_words.flatten(),
                           type_preds.view(num_types, -1), true_types.flatten(), masked_ids.flatten())

    sum(batch_losses).backward()
    optim.step()
    optim.zero_grad()

    return (batch_losses[0].item(), batch_losses[1].item()), sent_stats, type_stats


def train_batches(model: TypedLM, dl: LazyLoader, loss_fn: MixedLoss, optim: Optimizer, num_batches: int,
                  device: str) -> Tuple[Tuple[float, float], float, float]:
    batch_idx, epoch_loss_mlm, epoch_loss_st = 0, 0., 0.

    sum_sent, sum_cor_sent, sum_words, sum_cor_words = 0, 0, 0, 0

    for batch_idx in range(num_batches):
        batch = dl.get_processed_batch()
        in_words, true_words, true_types, pad_mask, masked_ids = [tensor.to(device) for tensor in batch]
        batch_stats = train_batch(model, loss_fn, optim, in_words, pad_mask, true_words, true_types, masked_ids)

        batch_losses, (num_sent, num_cor_sent), (num_words, num_cor_words) = batch_stats
        epoch_loss_mlm += batch_losses[0]
        epoch_loss_st += batch_losses[1]
        sum_sent += num_sent
        sum_cor_sent += num_cor_sent
        sum_words += num_words
        sum_cor_words += num_cor_words

    epoch_loss_mlm /= (batch_idx+1)
    epoch_loss_st /= (batch_idx+1)

    return (epoch_loss_mlm, epoch_loss_st), sum_cor_sent/sum_sent, sum_cor_words/sum_words
