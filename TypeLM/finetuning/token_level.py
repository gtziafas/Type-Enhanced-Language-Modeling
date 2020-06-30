from TypeLM.preprocessing.tokenizer import Tokenizer, WordTokenizer
from typing import List, Tuple, Callable
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence as _pad_sequence
from torch import LongTensor, tensor, long, FloatTensor
from TypeLM.neural.model import TypedLM
from torch.nn import Module, Linear
from TypeLM.neural.training import type_accuracy as token_accuracy
from torch.optim import Optimizer


Sample = Tuple[List[int], List[int]]
Samples = List[Sample]


def tokenize_data(tokenizer: Tokenizer, data: List[List[Tuple[str, int]]], pad: int) -> Samples:
    unzipped = [list(zip(*datum)) for datum in data]
    wss, tss = list(zip(*unzipped))
    return [tokenize_token_pairs(tokenizer.word_tokenizer, ws, ts, pad) for ws, ts in zip(wss, tss)]


def tokenize_token_pairs(wtokenizer: WordTokenizer, words: List[str], tokens: List[int], pad: int) -> Sample:
    assert len(words) == len(tokens)
    words = [wtokenizer.core.tokenize(w) for w in words]
    tokens = [[t] + [pad] * (len(w) - 1) for w, t in zip(words, tokens)]
    words = sum(words, [])
    tokens = sum(tokens, [])
    assert len(words) == len(tokens)
    word_ids = wtokenizer.core.convert_tokens_to_ids(words)
    return [wtokenizer.core.cls_token_id] + word_ids + [wtokenizer.core.sep_token_id], [pad] + tokens + [pad]


class TokenDataset(Dataset):
    def __init__(self, data: Samples):
        self._data = data

    @property
    def data(self) -> Samples:
        return self._data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item: int) -> Sample:
        return self.data[item]


def token_collator(word_pad: int, token_pad: int) -> Callable[[Samples], Tuple[LongTensor, LongTensor]]:
    def collate_fn(samples: Samples) -> Tuple[LongTensor, LongTensor]:
        xs, ys = list(zip(*samples))
        return (_pad_sequence([tensor(x, dtype=long) for x in xs], batch_first=True, padding_value=word_pad),
                _pad_sequence([tensor(y, dtype=long) for y in ys], batch_first=True, padding_value=token_pad))
    return collate_fn


class TypedLMForTokenClassification(Module):
    def __init__(self, model_maker: Callable[[], TypedLM], num_classes: int):
        super(TypedLMForTokenClassification, self).__init__()
        self.core = model_maker()
        self.token_classifier = Linear(in_features=self.core.word_embedder.embedding_dim, out_features=num_classes)

    def forward(self, words: LongTensor, pad_mask: LongTensor) -> FloatTensor:
        deep = self.core.encode(words, pad_mask)[1]
        return self.token_classifier(deep)


def train_batch(model: TypedLMForTokenClassification, loss_fn: Module, optim: Optimizer, words: LongTensor,
                padding_mask: LongTensor, tokens: LongTensor, token_pad: int) -> Tuple[float, Tuple[int, int]]:
    model.train()

    num_tokens = words.shape[0] * words.shape[1]
    predictions = model.forward(words, padding_mask)
    _, token_stats = token_accuracy(predictions.argmax(dim=-1), tokens, token_pad)

    batch_loss = loss_fn(predictions.view(num_tokens, -1), tokens.flatten())
    batch_loss.backward()
    optim.step()
    optim.zero_grad()
    return batch_loss.item(), token_stats


def train_epoch(model: TypedLMForTokenClassification, loss_fn: Module, optim: Optimizer,
                dataloader: DataLoader, token_pad: int, word_pad: int, device: str) -> Tuple[float, float]:
    epoch_loss, sum_tokens, sum_correct_tokens = 0., 0, 0

    for words, tokens in dataloader:
        padding_mask = (words != word_pad).unsqueeze(1).repeat(1, words.shape[1], 1).long().to(device)
        words = words.to(device)
        tokens = tokens.to(device)
        loss, (batch_total, batch_correct) = train_batch(model, loss_fn, optim, words, padding_mask, tokens, token_pad)
        sum_tokens += batch_total
        sum_correct_tokens += batch_correct
        epoch_loss += loss
    return epoch_loss / len(dataloader), sum_correct_tokens / sum_tokens


# todo
def eval_batch(*args, **kwargs):
    pass


# todo
def eval_epoch(*args, **kwargs):
    pass

