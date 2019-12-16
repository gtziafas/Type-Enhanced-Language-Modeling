from TypeLM.utils.imports import * 
from torch.optim import Optimizer 
from torch.utils.data import DataLoader 

def train_batch(model: Module, words_batch_inp: LongTensor, words_batch_truth: LongTensor, types_batch: LongTensor, 
			pad_mask: LongTensor, mask_off: LongTensor, loss_fn: tensor_map, optimizer: Optimizer) -> float:
	# total num of samples
	num_samples = words_batch.shape[0]  * words_batch.shape[1]

	# init optimizer
	optimizer.zero_grad()

	# forward pass and loss
	word_preds_batch, type_preds_batch = model(words_batch, pad_mask).view(num_samples, -1)
	batch_loss = loss_fn(word_preds_batch, types_batch, mask_off)

	# backprop
	batch_loss.backward()
	optimizer.step()

	return batch_loss.item()

def train_epoch(model: Module, dl: DataLoader, loss_fn: tensor_map, optimizer: Optimizer, device: str) -> float:
	epoch_loss = 0.
	for batch_idx, (words_batch, types_batch) in enumerate(dl):
		# to CUDA if applicable
		words_batch = words_batch.to(device)
		types_batch = types_batch.to(device)

		# train batch 
		batch_loss = train_batch(model, words_batch, types_batch, loss_fn, optimizer)
		epoch_loss += batch_loss
	# average epoch loss 
	epoch_loss /= batch_idx

	return epoch_loss

def train_model(model: Module, dl: DataLoader, loss_fn: tensor_map, optimizer: Optimizer, num_epochs: int, device: str) -> :
	print('\nStarted training')
	losses = []
	for epoch in range(num_epochs):
		train_loss = train_epoch(model, dl, loss_fn, optimizer, device)
		losses.append(train_loss)
		print('Epoch {:d}/{:d}, training loss={:.4f}'.format(epoch+1, num_epochs, train_loss))
	print('\nFinished training')

	return losses 


