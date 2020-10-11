from TypeLM.preprocessing.defaults import default_tokenizer
from nlp_nl.nl_eval.datasets import create_diedat
from TypeLM.finetuning.token_level import (tokenize_data, TokenDataset, DataLoader, CrossEntropyLoss, tensor,
                                           TypedLMForTokenClassification, default_pretrained, _pad_sequence,
                                           Samples, Tensor, LongTensor, Module, token_accuracy)
from torch.optim import AdamW, Optimizer
from typing import List, Dict, Tuple, Callable
import sys


def mask_from_tokens(preds: Tensor, tokens: LongTensor) -> Tuple[Tensor, LongTensor]:
    return preds[tokens>0].unsqueeze(0), tokens[tokens>0].unsqueeze(0)


def train_batch(model: TypedLMForTokenClassification, loss_fn: Module, optim: Optimizer, words: LongTensor,
                padding_mask: LongTensor, tokens: LongTensor, token_pad: int) -> Tuple[float, Tuple[int, int]]:
    model.train()

    predictions = model.forward(words, padding_mask)
    predictions, tokens = mask_from_tokens(predictions, tokens)

    _, token_stats = token_accuracy(predictions.argmax(dim=-1), tokens, token_pad)

    batch_loss = loss_fn(predictions.flatten(), tokens.flatten())
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


@no_grad()
def eval_batch(model: TypedLMForTokenClassification, loss_fn: Module, words: LongTensor,
               padding_mask: LongTensor, tokens: LongTensor, token_pad: int) \
        -> Tuple[float, Tuple[int, int], List[List[int]]]:
    model.eval()

    predictions = model.forward(words, padding_mask)
    predictions, tokens = mask_from_tokens(predictions, tokens)
    predictions_sharp = predictions.argmax(dim=-1)
    _, token_stats = token_accuracy(predictions_sharp, tokens, token_pad)

    batch_loss = loss_fn(predictions.flatten(), tokens.flatten())
    return batch_loss.item(), token_stats, predictions_sharp.tolist()



def eval_epoch(model: TypedLMForTokenClassification, loss_fn: Module, dataloader: DataLoader, token_pad: int,
               word_pad: int, device: str) -> Tuple[float, float]:
    epoch_loss, sum_tokens, sum_correct_tokens = 0., 0, 0
    predictions: List[List[int]] = []

    for words, tokens in dataloader:
        padding_mask = (words != word_pad).unsqueeze(1).repeat(1, words.shape[1], 1).long().to(device)
        words = words.to(device)
        tokens = tokens.to(device)
        loss, (batch_total, batch_correct), preds = eval_batch(model, loss_fn, words, padding_mask, tokens, token_pad)
        sum_tokens += batch_total
        sum_correct_tokens += batch_correct
        epoch_loss += loss
    return epoch_loss / len(dataloader), sum_correct_tokens / sum_tokens


def token_collator(word_pad: int, token_pad: int, mask_token: int) -> Callable[[Samples], Tuple[LongTensor, LongTensor]]:
    def collate_fn(samples: Samples) -> Tuple[LongTensor, LongTensor]:
        xs, ys = list(zip(*samples))
        xs = [[mask_token if y[j]>0 else w for j,w in enumerate(x)] for x,y in samples]
        return (_pad_sequence([tensor(x, dtype=long) for x in xs], batch_first=True, padding_value=word_pad),
                _pad_sequence([tensor(y, dtype=long) for y in ys], batch_first=True, padding_value=token_pad))
    return collate_fn


def main(diedat_path: str, model_path: str, device: str, batch_size_train: int, batch_size_dev: int,
         num_epochs: int) -> None:
    def sprint(s: str) -> None:
        print(s)
        sys.stdout.flush()

    tokenizer = default_tokenizer()
    mask_token_id = tokenizer.word_tokenizer.core.mask_token_id
    word_pad_id = tokenizer.word_tokenizer.core.pad_token_id
    token_pad_id = -100
    diedat = create_diedat(diedat_path)
    offset = 0
    loss_fn = CrossEntropyLoss(ignore_index=token_pad_id, reduction='mean')

    processed_train = tokenize_data(tokenizer, [t for t in diedat.train_data if len(t) <= 100], \
        token_pad_id, offset)
    processed_dev = tokenize_data(tokenizer, [t for t in diedat.dev_data if len(t) <= 100], \
        token_pad_id, offset)
    processed_test = tokenize_data(tokenizer, [t for t in diedat.test_data if len(t) <= 100], \
        token_pad_id, offset)

    train_loader = DataLoader(dataset=TokenDataset(processed_train), batch_size=batch_size_train, shuffle=True,
                              collate_fn=token_collator(word_pad_id, token_pad_id, mask_token_id))
    dev_loader = DataLoader(dataset=TokenDataset(processed_dev), batch_size=batch_size_dev, shuffle=False,
                            collate_fn=token_collator(word_pad_id, token_pad_id, mask_token_id))
    test_loader = DataLoader(dataset=TokenDataset(processed_test), batch_size=batch_size_dev, shuffle=False,
                             collate_fn=token_collator(word_pad_id, token_pad_id, mask_token_id))

    model = TypedLMForTokenClassification(default_pretrained(model_path), len(diedat.class_map)).to(device)
    optim = AdamW(model.parameters(), lr=3e-05)

    sprint('Done with tokenization/loading, starting to train...')
    for epoch in range(num_epochs):
        train_loss, train_accu = train_epoch(model, loss_fn, optim, train_loader, token_pad_id, word_pad_id, device)
        sprint(f'Train loss:\t\t{train_loss}')
        sprint(f'Train accu:\t\t{train_accu}')
        val_loss, val_accu = eval_epoch(model, loss_fn, dev_loader, token_pad_id, word_pad_id, device)
        sprint(f'Dev loss:\t\t{val_loss}')
        sprint(f'Dev accu:\t\t{val_accu}')
        test_loss, test_accu = eval_epoch(model, loss_fn, test_loader, token_pad_id, word_pad_id, device)
        sprint(f'Dev loss:\t\t{test_loss}')
        sprint(f'Dev accu:\t\t{test_accu}')


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-p', '--diedat_path', help='Path to diedat folder.')
    parser.add_argument('-m', '--model_path', help='Path to pretrained model')
    parser.add_argument('-d', '--device', help='Which device to use', default='cuda')
    parser.add_argument('-b', '--batch_size_train', help='Training batch size', default=32, type=int)
    parser.add_argument('-bd', '--batch_size_dev', help='Validation batch size', default=512, type=int)
    parser.add_argument('-e', '--num_epochs', help='How many epochs to train for', default=10, type=int)

    kwargs = vars(parser.parse_args())

    main(**kwargs)
