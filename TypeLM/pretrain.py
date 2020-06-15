from TypeLM.neural.defaults import *
from TypeLM.neural.training import *
from TypeLM.preprocessing.defaults import *
from typing import Optional
import sys

_sents_in_dset = 60290072
_batch_size = 2
_num_batches_in_dset = _sents_in_dset // _batch_size
_num_subepochs_per_epoch = 1000000
_num_batches_per_subepoch = _num_batches_in_dset // _num_subepochs_per_epoch
_warmup_subepochs = 100
_warmup_steps = _warmup_subepochs * _num_batches_per_subepoch
_device = 'cuda'

print(f'Training a dataset of {_sents_in_dset} samples with a batch size of {_batch_size}.')
print(f'Reporting averages every {_num_batches_per_subepoch * _batch_size} samples.')

loader = default_dataloader(path='./TypeLM/data/indexing/atomic_dump', chunk_size=1024000, batch_size=_batch_size)
model = default_model().to('cuda')
loss_fn = default_loss()
optim = default_optimizer(model, warmup_steps=_warmup_steps)


def start(save_path: str):
    resume(epoch=0, save_path=save_path, load_path=None)


def resume(epoch: int, save_path: str, load_path: Optional[str]):
    if epoch != 0:
        tmp = torch.load(load_path)
        model.load_state_dict(tmp['model_state_dict'])
        optim.opt.load_state_dict(tmp['opt'])
        optim.step_num = epoch * _num_batches_in_dset

    print('=' * 64)
    print(f'EPOCH {epoch}')
    print('=' * 64)
    print('\n\n')
    for subepoch in range(_num_subepochs_per_epoch):
        tmp = train_batches(model, loader, loss_fn, optim, _num_batches_per_subepoch, _device)
        (mlm_loss, st_loss), s_acc, atom_acc = tmp
        print('-' * 64)
        print(f'Subepoch\t\t{subepoch}')
        print(f'\tMLM Loss:\t\t{mlm_loss}')
        print(f'\tST Loss:\t\t{st_loss}')
        print(f'\tSentence acc:\t{s_acc}')
        print(f'\tAtom acc:\t\t{atom_acc}')
        print(f'\tCurrent lr:\t\t{optim.lr}')
        sys.stdout.flush()
    print('Finished training epoch.')
    torch.save({'model_state_dict': model.state_dict(), 'opt': optim.opt.state_dict()}, save_path)
    print('Saved model')
