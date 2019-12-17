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

    def post_processor(sentences: Samples) -> Tuple[LongTensor, LongTensor, LongTensor, LongTensor, LongTensor]:
        true_words, types = list(zip(*sentences))
        masked_words, masked_indices = list(zip(*list(map(masker, true_words))))
        masked_words = pad_sequence(list(map(LongTensor, masked_words)))
        true_words = pad_sequence(list(map(LongTensor, true_words)))
        types = pad_sequence(list(map(LongTensor, types)))
        masked_indices = pad_sequence(list(map(LongTensor, masked_indices)))
        word_pads = pad_sequence([torch.ones(len(sentence)) for sentence in true_words])
        return masked_words, true_words, word_pads, types, masked_indices

    return DataLoader('./shit_5-1.txt', 10, 2, post_processor)


def train_batch(model: TypeFactoredLM, masked_words: LongTensor, true_words: LongTensor, types: LongTensor,
                pad: LongTensor, masked_indices: LongTensor, loss_fn: MixedLoss,
                optimizer: Optimizer) -> float:

    # total num of samples
    num_samples = masked_words.shape[0] * masked_words.shape[1]

    # forward pass and loss
    word_preds, type_preds = model(masked_words, pad)
    batch_loss = loss_fn(word_preds.view(num_samples, -1), true_words.flatten(),
                         type_preds.view(num_samples, -1), types.flatten(), masked_indices)

    # backprop
    batch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return batch_loss.item()


def train_batches(model: TypeFactoredLM, dl: DataLoader, loss_fn: MixedLoss, optimizer: Optimizer, num_batches: int,
                  device: str) -> float:
    batch_idx, epoch_loss = 0, 0.

    for batch_idx in range(num_batches):
        batch = dl.get_processed_batch()
        words_input, words_truth, types, pad, words_mask = list(map(lambda x: x.to(device), batch))
        batch_loss = train_batch(model, words_input, words_truth, types, pad, words_mask, loss_fn, optimizer)
        epoch_loss += batch_loss
    epoch_loss /= (batch_idx+1)

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


