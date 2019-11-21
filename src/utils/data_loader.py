from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from src.utils.imports import *
from src.utils.utils import mask_indices, mask_sampling
from torch.nn.utils.rnn import pad_sequence


Sample = Sequence[Tuple[str, str]]
Samples = Sequence[Sample]


def mask_sample(sample: Sample, random_chance: float = 0.15, mask_token: str = '[MASK]') -> Tuple[Sample, ints]:
    words, types = list(zip(*sample))
    masked_indices = mask_indices(len(words), random_chance)
    # todo. actual sampling
    words = mask_sampling(words, masked_indices, lambda: mask_token)
    types = mask_sampling(types, masked_indices, lambda: mask_token)
    return list(zip(words, types)), masked_indices


class LazyDataset(Dataset):
    def __init__(self):
        super(LazyDataset, self).__init__()
        pass

    def __getitem__(self, item) -> Sample:
        # todo. lazily evaluate an item
        pass


def convert_indices_to_bool_masks(masks: Sequence[LongTensor]) -> LongTensor:
    """
        take a list of B LongTensors indexing masked elements
        return a boolean matrix OUT of size B x S where S the max seq len,
            where OUT[b, s] = 1 IFF s in masks[b]
                                else 0
    """
    # init a zero tensor
    # set tensor[idx] = 1 where idx each idx specified by in


