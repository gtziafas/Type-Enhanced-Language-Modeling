from TypeLM.neural.defaults import *
from TypeLM.neural.training import *
from TypeLM.preprocessing.defaults import *
from typing import Optional
import sys

_sents_in_dset = 60290072
_batch_size = 128
_num_batches_in_dset = _sents_in_dset // _batch_size
_num_subepochs_per_epoch = 100
_num_batches_per_subepoch = _num_batches_in_dset // _num_subepochs_per_epoch
# _warmup_subepochs = 100
_warmup_steps = 1e04
_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'Training a dataset of {_sents_in_dset} samples with a batch size of {_batch_size}.')
print(f'Reporting averages every {_num_batches_per_subepoch * _batch_size} samples.')

loader = default_dataloader(path='/data/s3913171/Lassy-Large/atomic_dump_small', chunk_size=1024000,
                            batch_size=_batch_size)
model = default_model().to('cuda')
loss_fn = default_loss()
optim = default_optimizer(model, warmup_steps=int(_warmup_steps))


def sprint(in_str: str) -> None:
    print(in_str)
    sys.stdout.flush()


def start(save_path: str):
    resume(epoch=0, save_path=save_path, load_path=None)


def resume(epoch: int, save_path: str, load_path: Optional[str]):
    if epoch != 0:
        tmp = torch.load(load_path)
        model.load_state_dict(tmp['model_state_dict'])
        optim.opt.load_state_dict(tmp['opt'])
        optim.step_num = epoch * _num_batches_in_dset

    sprint('=' * 64)
    sprint(f'EPOCH {epoch}')
    sprint('=' * 64)
    sprint('\n\n')
    for subepoch in range(_num_subepochs_per_epoch):
        tmp = train_batches(model, loader, loss_fn, optim, _num_batches_per_subepoch, _device)
        (mlm_loss, st_loss), s_acc, atom_acc = tmp
        sprint('-' * 64)
        sprint(f'Subepoch\t\t{subepoch}')
        sprint(f'\tMLM Loss:\t\t{mlm_loss}')
        sprint(f'\tST Loss:\t\t{st_loss}')
        sprint(f'\tSentence acc:\t{s_acc}')
        sprint(f'\tAtom acc:\t\t{atom_acc}')
        sprint(f'\tCurrent lr:\t\t{optim.lr}')

    sprint('Finished training epoch.')
    torch.save({'model_state_dict': model.state_dict(), 'opt': optim.opt.state_dict()}, save_path)
    sprint('Saved model')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--save_path', help='where to save the model once training ends', type=str)
    parser.add_argument('-l', '--load_path', help='where to load the model from for resuming training', type=Optional[str], default=None)
    parser.add_argument('-e', '--epoch', help='which epoch to resume training from', type=Optional[int], default=None)

    kwargs = vars(parser.parse_args())
    if kwargs['epoch'] is None:
        start(kwargs['save_path'])
    else:
        resume(**kwargs)
