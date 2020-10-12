from TypeLM.preprocessing.tokenizer import Tokenizer, WordTokenizer
from typing import List, Tuple, Callable
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence as _pad_sequence
from torch import LongTensor, Tensor, tensor, long, no_grad, stack, load
from TypeLM.neural.model import TypedLM
from TypeLM.neural.defaults import default_model
from torch.nn import Module, Linear, CrossEntropyLoss, Dropout
from torch.optim import Optimizer


Sample = Tuple[List[str], int]
Samples = List[Sample]


def vanilla_accuracy(predictions: Tensor, truths: LongTensor) -> float:
    return (predictions.argmax(dim=-1) == truths).sum().item() / predictions.shape[0]


def tokenize_data(tokenizer: Tokenizer, data: Samples) -> List[Tuple[List[int], int]]:
    return [(tokenize_words(tokenizer.word_tokenizer, words), tag) for words, tag in zip(*data)]


def tokenize_words(wtokenizer: WordTokenizer, words: List[int], tag: int) -> List[int]:
    words = [wtokenizer.core.tokenize(w) for w in words]
    word_ids = wtokenizer.core.convert_tokens_to_ids(words)
    _cls = [wtokenizer.core.cls_token_id]
    _sep = [wtokenizer.core.sep_token_id]
    return _cls + word_ids + _sep


class SequenceDataset(Dataset):
    def __init__(self, data: Samples):
        self._data = data

    @property
    def data(self) -> Samples:
        return self._data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item: int) -> Sample:
        return self.data[item]


class TypedLMForSequenceClassification(Module):
    def __init__(self, model_maker: Callable[[], TypedLM], num_classes: int, dropout_rate: float = 0.1):
        super(TypedLMForSequenceClassification, self).__init__()
        self.core = model_maker()
        self.classifier = Linear(in_features=self.core.word_embedder.embedding_dim, out_features=num_classes)
        self.dropout = Dropout(dropout_rate)

    def forward(self, words: LongTensor, pad_mask: LongTensor) -> Tensor:
        deep = self.core.encode(words, pad_mask)[1]
        return self.classifier(self.dropout(deep[:,0,:]))


def default_pretrained(path: str) -> Callable[[], TypedLM]:
    def model_maker() -> TypedLM:
        model = default_model()
        tmp = load(path)
        model.load_state_dict(tmp['model_state_dict'])
        return model
    return model_maker


def sequence_collator(word_pad: int) -> Callable[[Samples], Tuple[LongTensor, LongTensor]]:
    def collate_fn(samples: Samples) -> Tuple[LongTensor, LongTensor]:
        xs, ys = list(zip(*samples))
        return (_pad_sequence([tensor(x, dtype=long) for x in xs], batch_first=True, padding_value=word_pad),
                stack([tensor(y, dtype=long) for y in ys], dim=0))
    return collate_fn


def train_batch(model: TypedLMForSequenceClassification, loss_fn: Module, optim: Optimizer, words: LongTensor,
                padding_mask: LongTensor, tokens: LongTensor) -> Tuple[float, float]:
    model.train()

    predictions = model.forward(words, padding_mask)
    accuracy = vanilla_accuracy(predictions, tokens)
    batch_loss = loss_fn(predictions, tokens)
    
    batch_loss.backward()
    optim.step()
    optim.zero_grad()
    return batch_loss.item(), token_stats


def train_epoch(model: TypedLMForSequenceClassification, loss_fn: Module, optim: Optimizer,
                dataloader: DataLoader, token_pad: int, word_pad: int, device: str) -> Tuple[float, float]:
    epoch_loss, epoch_accuracy = 0., 0.

    for words, tokens in dataloader:
        padding_mask = (words != word_pad).unsqueeze(1).repeat(1, words.shape[1], 1).long().to(device)
        words = words.to(device)
        tokens = tokens.to(device)
        loss, accuracy = train_batch(model, loss_fn, optim, words, padding_mask, tokens)
        epoch_loss += loss
        epoch_accuracy += accuracy
    return epoch_loss / len(dataloader), epoch_accuracy / len(dataloader)


@no_grad()
def eval_batch(model: TypedLMForSequenceClassification, loss_fn: Module, words: LongTensor,
               padding_mask: LongTensor, tokens: LongTensor) \
        -> Tuple[float, float]:
    model.eval()

    num_tokens = words.shape[0] * words.shape[1]
    predictions = model.forward(words, padding_mask)
    accuracy = vanilla_accuracy(predictions, tokens)

    batch_loss = loss_fn(predictions.view(num_tokens, -1), tokens.flatten())
    return batch_loss.item(), accuracy


def eval_epoch(model: TypedLMForSequenceClassification, loss_fn: Module, dataloader: DataLoader, \
                word_pad: int, device: str) -> Tuple[float, float, List[List[int]]]:
    epoch_loss, epoch_accuracy = 0., 0.

    for words, tokens in dataloader:
        padding_mask = (words != word_pad).unsqueeze(1).repeat(1, words.shape[1], 1).long().to(device)
        words = words.to(device)
        tokens = tokens.to(device)
        loss, accuracy = eval_batch(model, loss_fn, words, padding_mask, tokens, token_pad)
        epoch_loss += loss
        epoch_accuracy += accuracy
    return epoch_loss / len(dataloader), epoch_accuracy / len(dataloader)
