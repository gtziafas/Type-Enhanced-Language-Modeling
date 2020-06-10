from TypeLM.neural.defaults import *
from TypeLM.neural.training import *
from TypeLM.preprocessing.defaults import *

batch_size = 32

loader = default_dataloader(path='./TypeLM/data/indexing/atomic_dump', chunk_size=20480, batch_size=16)
model = default_model().to('cuda')
loss_fn = default_loss()
optim = default_optimizer(model, warmup_steps=4096)

train_batches(model, loader, loss_fn, optim, 1024, 'cuda')