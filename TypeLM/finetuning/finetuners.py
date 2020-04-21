from TypeLM.model.type_factored_lm import TypeFactoredLM
from TypeLM.utils.imports import *
from TypeLM.utils.utils import type_accuracy as token_accuracy

from torch import load
from torch.utils.data import DataLoader

from itertools import chain

from torch.optim import AdamW
from torch.nn import CrossEntropyLoss


def expand_mask(pad_mask: LongTensor) -> LongTensor:
    assert len(pad_mask.shape) == 2
    return pad_mask.unsqueeze(-1).repeat(1, 1, pad_mask.shape[1])


def get_eos_features(token_features: Tensor, pad_mask: LongTensor) -> Tensor:
    assert len(pad_mask.shape) == 2
    assert len(token_features.shape) == 3
    last_positions = pad_mask.sum(dim=-1) - 1
    return torch.stack([token_features[i, last_positions[i]] for i in range(last_positions.shape[0])])


class Finetuner(Module):
    def __init__(self, core_maker: Callable[[], TypeFactoredLM], path: str, num_classes: int, dropout_rate: float = 0.1):
        super(Finetuner, self).__init__()
        self.core = core_maker()
        checkpoint = load(path)
        self.core.load_state_dict(checkpoint['model_state_dict'])
        self.dropout = Dropout(dropout_rate)
        self.num_classes = num_classes

    def forward(self, *args):
        pass

    def train_batch(self, batch_x: LongTensor, pad_mask: LongTensor, batch_y: LongTensor,
                    optimizer: Optimizer, loss_fn: Module) -> Tuple[float, Tuple[int, int]]:
        self.train()

        batch_p = self.forward(batch_x, pad_mask)[:, :-1]
        loss = loss_fn(batch_p.view(-1, self.num_classes), batch_y.flatten())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        token_stats = token_accuracy(predictions=batch_p.argmax(dim=-1), truth=batch_y, ignore_idx=0)[1]

        return loss.item(), (token_stats[0], token_stats[1])

    def train_epoch(self, dataloader: DataLoader, optimizer: Optimizer, loss_fn: Module) -> Tuple[float, float]:
        loss, correct, total = 0., 0, 0
        for batch_x, pad_mask, batch_y in dataloader:
            batch_loss, (batch_correct, batch_total) = self.train_batch(batch_x, pad_mask, batch_y, optimizer, loss_fn)
            loss += batch_loss
            correct += batch_correct
            total += batch_total
        return loss, correct/total

    @torch.no_grad()
    def eval_batch(self, batch_x: LongTensor, pad_mask: LongTensor, batch_y: LongTensor,
                   loss_fn: Module) -> Tuple[float, Tuple[int, int]]:
        self.eval()

        batch_p = self.forward(batch_x, pad_mask)[:, :-1]
        loss = loss_fn(batch_p.view(-1, self.num_classes), batch_y.flatten())

        token_stats = token_accuracy(predictions=batch_p.argmax(dim=-1), truth=batch_y, ignore_idx=0)[1]

        return loss.item(), (token_stats[0], token_stats[1])

    def eval_epoch(self, dataloader: DataLoader, loss_fn: Module) -> Tuple[float, float]:
        loss, correct, total = 0., 0, 0
        for batch_x, pad_mask, batch_y in dataloader:
            batch_loss, (batch_correct, batch_total) = self.eval_batch(batch_x, pad_mask, batch_y, loss_fn)
            loss += batch_loss
            correct += batch_correct
            total += batch_total
        return loss, correct / total


class TokenClassification(Finetuner):
    def __init__(self, core_maker: Callable[[], TypeFactoredLM], path: str, num_classes: int):
        super(TokenClassification, self).__init__(core_maker, path, num_classes)
        self.token_out = Linear(in_features=self.core.d_model, out_features=self.num_classes, bias=True)

    def forward(self, x: LongTensor, pad_mask: LongTensor) -> Tensor:
        token_features = self.core.get_token_features(word_ids=x, pad_mask=expand_mask(pad_mask))
        token_features = self.dropout(token_features)
        return self.token_out(token_features)

    @torch.no_grad()
    def infer_batch(self, batch_x: LongTensor, pad_mask: LongTensor) -> List[List[int]]:
        self.eval()
        batch_p: List[List[int]]
        batch_p = self.forward(batch_x, pad_mask).argmax(dim=-1).tolist()
        pad_mask = pad_mask.tolist()

        return [[prediction for i, prediction in enumerate(sentence) if pad_mask[j][i] != 0]
                for j, sentence in enumerate(batch_p)]

    def infer_dataloader(self, dataloader: DataLoader) -> List[List[int]]:
        return list(chain.from_iterable([self.infer_batch(batch_x, pad_mask) for batch_x, pad_mask in dataloader]))


class SequenceClassification(Finetuner):
    def __init__(self, core_maker: Callable[[], TypeFactoredLM], path: str, num_classes: int):
        super(SequenceClassification, self).__init__(core_maker, path, num_classes)
        self.sequence_out = Linear(in_features=self.core.d_model, out_features=self.num_classes, bias=True)

    def forward(self, batch_x: LongTensor, pad_mask: LongTensor) -> Tensor:
        contextualized = self.core.get_token_features(batch_x, pad_mask)
        eos = get_eos_features(contextualized, pad_mask)
        eos = self.dropout(eos)
        return self.sequence_out(eos)

    @torch.no_grad()
    def infer_batch(self, batch_x: LongTensor, pad_mask: LongTensor) -> List[int]:
        self.eval()
        return self.forward(batch_x, pad_mask).argmax(dim=-1).tolist()

    def infer_dataloader(self, dataloader: DataLoader) -> List[int]:
        return list(chain.from_iterable([self.infer_batch(batch_x, pad_mask) for batch_x, pad_mask in dataloader]))
