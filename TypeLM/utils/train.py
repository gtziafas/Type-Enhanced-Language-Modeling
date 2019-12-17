from TypeLM.utils.imports import * 
from TypeLM.model.loss import MixedLoss
from TypeLM.model.type_factored_lm import TypeFactoredLM
from TypeLM.data.loader import DataLoader
from TypeLM.data.masker import default_masker

from torch.optim import Optimizer

from torch.nn.utils.rnn import pad_sequence as _pad_sequence


def pad_sequence(x):
    return _pad_sequence(x, batch_first=True, padding_value=0)


def default_dataloader():
    masker = default_masker()

    def post_processor(sentences: Samples) -> Tuple[LongTensor, LongTensor, LongTensor, LongTensor]:
        # todo. pad shit
        true_words, types = list(zip(*sentences))
        masked_words, masked_indices = list(zip(*list(map(masker, true_words))))
        masked_words = pad_sequence(list(map(LongTensor, masked_words)))
        true_words = pad_sequence(list(map(LongTensor, true_words)))
        types = pad_sequence(list(map(LongTensor, types)))
        masked_indices = pad_sequence(list(map(LongTensor, masked_indices)))
        return masked_words, true_words, types, masked_indices

    return DataLoader('./shit_5-1.txt', 10, 2, post_processor)


def train_batch(model: TypeFactoredLM, masked_words: LongTensor, true_words: LongTensor, types: LongTensor,
                pad: LongTensor, masked_indices: LongTensor, loss_fn: MixedLoss,
                optimizer: Optimizer) -> float:

    # total num of samples
    num_samples = masked_words.shape[0] * masked_words.shape[1]

    # forward pass and loss
    word_preds_batch, type_preds_batch = model(masked_words, pad)
    batch_loss = loss_fn(word_preds_batch.view(num_samples,-1), true_words.flatten(),
                         type_preds_batch.view(num_samples, -1), types.flatten(), masked_indices)

    # backprop
    batch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return batch_loss.item()


def train_nbatches(model: TypeFactoredLM, dl: DataLoader, loss_fn: MixedLoss, optimizer: Optimizer, nbatches: int,
                   device: str) -> float:
    epoch_loss = 0.

    for batch_idx in range(nbatches):
        batch = dl.get_processed_batch()
        words_input, words_truth, types, pad, words_mask = list(map(lambda x: x.to(device), batch))
        batch_loss = train_batch(model, words_input, words_truth, types, pad, words_mask, loss_fn, optimizer)
        epoch_loss += batch_loss
    epoch_loss /= batch_idx

    return epoch_loss


# def train_model(model: TypeFactoredLM, dl: DataLoader, loss_fn: MixedLoss, optimizer: Optimizer,
# num_epochs: int, device: str) -> :
# 	print('\nStarted training')
# 	losses = []
# 	for epoch in range(num_epochs):
# 		train_loss = train_nbatches(model, dl, loss_fn, optimizer, device)
# 		losses.append(train_loss)
# 		print('Epoch {:d}/{:d}, training loss={:.4f}'.format(epoch+1, num_epochs, train_loss))
# 	print('\nFinished training')

# 	return losses 


def eval_batch(model: TypeFactoredLM, masked_words: LongTensor, true_words: LongTensor, types: LongTensor,
                pad: LongTensor, masked_indices: LongTensor, loss_fn: MixedLoss) -> float:
    model.eval()

    num_samples = X_batch.shape[0] * X_batch.shape[1]

    word_preds_batch, type_preds_batch = model(masked_words, pad)
    batch_loss = loss_fn(word_preds_batch.view(num_samples,-1), true_words.flatten(),
                        type_preds_batch.view(num_samples, -1), types.flatten(), masked_indices)

    return batch_loss.item()


def eval_nbatches(model: TypeFactoredLM, dl: DataLoader, loss_fn: MixedLoss, nbatches: int, device: str) -> float:
    epoch_loss = 0.

    for batch_idx in range(nbatches):
        batch = dl.get_processed_batch()
        words_input, words_truth, types, pad, words_mask = list(map(lambda x: x.to(device), batch))
        batch_loss = eval_batch(model, words_input, words_truth, types, pad, words_mask, loss_fn)
        epoch_loss += batch_idx
    epoch_loss /= batch_idx
    model.train()
    
    return epoch_loss

def type_predictions_accuracy(type_preds: LongTensor, true_types: LongTensor, ignore_idx: int) -> Tuple[int, int]:
    correct_types = torch.ones_like(type_preds)
    correct_types[type_preds != true_types] = 0
    correct_types[true_types == ignore_idx] = 1 

    num_correct_types = correct_types.sum().item()
    num_masked_types = len(true_types[true_types == ignore_idx])

    # total number of non-ignored types 
    total = type_preds.shape[0] * type_preds.shape[1] - num_masked_types

    # total number of correctly predicted types 
    num_correct_types -= num_masked_types

    return total, num_correct_types


def measure_type_predictions_accuracy(model: TypeFactoredLM, dl: DataLoader, nbatches: int) -> float:
    correct = 0.
    total = 0.
    for batch_idx in range(nbatches):
        batch = dl.get_processed_batch()
        words_input, words_truth, types, pad, words_mask = list(map(lambda x: x.to(device), batch))
        word_preds_batch, type_preds_batch = model(masked_words, pad)
        local_total, local_correct = type_predictions_accuracy(type_preds_batch.argmax(dim=-1), types, ignore_idx=0)
        correct += local_correct
        total += local_total
    return correct / total