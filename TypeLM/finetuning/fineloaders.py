from torch.utils.data import DataLoader, Dataset
from TypeLM.utils.imports import *
from TypeLM.data.tokenizer import default_tokenizer, Indexer, EOS
from TypeLM.model.train import pad_sequence
from tqdm import tqdm

tokenizer = default_tokenizer()
indexer = Indexer(tokenizer)


def make_token_train_dl(samples: List[Tuple[List[str], List[int]]],
                        batch_size: int = 32, shuffle: bool = True, device: str = 'cpu') -> DataLoader:
    def my_collate(inps: List[Tuple[List[int], List[int]]]) -> Tuple[LongTensor, LongTensor, LongTensor]:
        lens = [len(inp[0]) for inp in inps]
        xs = pad_sequence([torch.tensor(inp[0], dtype=torch.long) for inp in inps])
        ys = pad_sequence([torch.tensor(inp[1], dtype=torch.long) for inp in inps])
        word_pads = torch.ones(xs.shape[0], xs.shape[1], xs.shape[1], dtype=torch.long)
        for i, l in enumerate(lens):
            word_pads[i, :, l::] = 0
        return (xs.to(device), word_pads.to(device), ys.to(device))

    sents, ids = list(zip(*samples))
    tokenized = list(map(lambda sent, id_list:
                         ([tokenizer.tokenize_word(word) for word in sent] + [EOS],
                         [class_label for class_label in id_list]),
                         sents, ids))
    indexed = [indexer.index_sentence(sample[0]) for sample in tokenized]
    dset = TokenTrain(indexed, [sample[1] for sample in tokenized])
    return DataLoader(dset, batch_size, shuffle=shuffle, collate_fn=my_collate)


def make_token_test_dl(samples: List[str], batch_size: int = 32, device: str = 'cpu') -> DataLoader:
    def my_collate(inps: List[List[int]]) -> Tuple[LongTensor, LongTensor]:
        lens = [len(inp) for inp in inps]
        xs = pad_sequence([torch.tensor(inp, dtype=torch.long) for inp in inps])
        word_pads = torch.ones(xs.shape[0], xs.shape[1], xs.shape[1], dtype=torch.long)
        for i, l in enumerate(lens):
            word_pads[i, :, l::] = 0
        return (xs.to(device), word_pads.to(device))

    tokenized = list(map(lambda sent:
                         [tokenizer.tokenize_word(word) for word in sent] + [EOS],
                         samples))
    indexed = [indexer.index_sentence(sample) for sample in tokenized]
    dset = TokenTest(indexed)
    return DataLoader(dset, batch_size, shuffle=False, collate_fn=my_collate)


class TokenTrain(Dataset):
    def __init__(self, idxed_sents: List[List[int]], class_labels: List[List[int]]):
        super(TokenTrain, self).__init__()
        self.xs = idxed_sents
        self.ys = class_labels

    def __len__(self) -> int:
        return len(self.xs)

    def __getitem__(self, item: int) -> Tuple[List[int], List[int]]:
        return self.xs[item], self.ys[item]


class TokenTest(Dataset):
    def __init__(self, idxed_sents: List[List[int]]):
        super(TokenTest, self).__init__()
        self.xs = idxed_sents

    def __len__(self) -> int:
        return len(self.xs)

    def __getitem__(self, item: int) -> List[int]:
        return self.xs[item]
