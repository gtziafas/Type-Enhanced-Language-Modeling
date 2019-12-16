from TypeLM.utils.imports import * 
from TypeLM.model.loss import MixedLoss
from TypeLM.model.type_factored_lm import TypeFactoredLM
from TypeLM.data.loader import DataLoader

from torch.optim import Optimizer 


def train_batch(model: TypeFactoredLM, words_batch_inp: LongTensor, words_batch_truth: LongTensor, types_batch: LongTensor, 
			pad_mask: LongTensor, words_mask: LongTensor, loss_fn: MixedLoss, optimizer: Optimizer) -> float:
	# total num of samples
	num_samples = words_batch_inp.shape[0]  * words_batch_inp.shape[1]

	# forward pass and loss
	word_preds_batch, type_preds_batch = model(words_batch_inp, pad_mask)
	batch_loss = loss_fn(word_preds_batch.view(num_samples,-1), words_batch_truth.flatten(),
					 type_preds_batch.view(num_samples, -1), types_batch.flatten(), words_mask)

	# backprop
	batch_loss.backward()
	optimizer.step()
	optimizer.zero_grad()

	return batch_loss.item()

def train_nbatches(model: TypeFactoredLM, dl: DataLoader, loss_fn: MixedLoss, optimizer: Optimizer, nbatches: int, device: str) -> float:
	epoch_loss = 0.
	for batch_idx in range(nbatches):
			batch = dl.__next__()
			batch = list(map(lambda x: x.to(device), batch))
			batch_loss = train_batch(model, *batch, loss_fn, optimizer)
			epoch_loss += batch_loss
	epoch_loss /= batch_idx

	return epoch_loss

# def train_model(model: TypeFactoredLM, dl: DataLoader, loss_fn: MixedLoss, optimizer: Optimizer, num_epochs: int, device: str) -> :
# 	print('\nStarted training')
# 	losses = []
# 	for epoch in range(num_epochs):
# 		train_loss = train_nbatches(model, dl, loss_fn, optimizer, device)
# 		losses.append(train_loss)
# 		print('Epoch {:d}/{:d}, training loss={:.4f}'.format(epoch+1, num_epochs, train_loss))
# 	print('\nFinished training')

# 	return losses 


