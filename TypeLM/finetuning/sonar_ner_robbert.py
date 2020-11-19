from transformers import RobertaTokenizer, RobertaForTokenClassification
from nlp_nl.nl_eval.datasets import create_sonar_ner
from TypeLM.finetuning.token_level import (Module, TokenDataset, token_collator, DataLoader, CrossEntropyLoss)
from TypeLM.finetuning.conlleval import evaluate
from torch.optim import AdamW
from typing import List, Dict, Tuple
import sys
import os

from functools import reduce
from operator import add

Sample = Tuple[List[int], List[int]]
Samples = List[Sample]


def train_batch(model: RobertaForTokenClassification, loss_fn: Module, optim: Optimizer, words: LongTensor,
                padding_mask: LongTensor, tokens: LongTensor, token_pad: int) -> Tuple[float, Tuple[int, int]]:
    model.train()

    num_tokens = words.shape[0] * words.shape[1]
    predictions = model.forward(words, padding_mask)[0]
    _, token_stats = token_accuracy(predictions.argmax(dim=-1), tokens, token_pad)

    batch_loss = loss_fn(predictions.view(num_tokens, -1), tokens.flatten())
    batch_loss.backward()
    optim.step()
    optim.zero_grad()
    return batch_loss.item(), token_stats


def train_epoch(model: RobertaForTokenClassification, loss_fn: Module, optim: Optimizer,
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
def eval_batch(model: RobertaForTokenClassification, loss_fn: Module, words: LongTensor,
               padding_mask: LongTensor, tokens: LongTensor, token_pad: int) \
        -> Tuple[float, Tuple[int, int], List[List[int]]]:
    model.eval()

    num_tokens = words.shape[0] * words.shape[1]
    predictions = model.forward(words, padding_mask)[0]
    predictions_sharp = predictions.argmax(dim=-1)
    _, token_stats = token_accuracy(predictions_sharp, tokens, token_pad)

    batch_loss = loss_fn(predictions.view(num_tokens, -1), tokens.flatten())
    return batch_loss.item(), token_stats, predictions_sharp.tolist()


@no_grad()
def eval_epoch(model: RobertaForTokenClassification, loss_fn: Module, dataloader: DataLoader, token_pad: int,
               word_pad: int, device: str) -> Tuple[float, float, List[List[int]]]:
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
        predictions += preds
    return epoch_loss / len(dataloader), sum_correct_tokens / sum_tokens, predictions


def tokenize_data(tokenizer: RobertaTokenizer, data: List[List[Tuple[str, int]]], pad: int, offset: int = 0) -> Samples:
    unzipped = [list(zip(*datum)) for datum in data]
    wss, tss = list(zip(*unzipped))
    return [tokenize_token_pairs(tokenizer, ws, ts, pad, offset) for ws, ts in zip(wss, tss)]


def tokenize_token_pairs(wtokenizer: RobertaTokenizer, words: List[str], tokens: List[int], pad: int, offset: int) \
        -> Sample:
    assert len(words) == len(tokens)
    words = [wtokenizer.tokenize(w) for w in words]
    tokens = [[t - offset] + [pad] * (len(w) - 1) for w, t in zip(words, tokens)]
    words = sum(words, [])
    tokens = sum(tokens, [])
    assert len(words) == len(tokens)
    word_ids = wtokenizer.convert_tokens_to_ids(words)
    return [wtokenizer.cls_token_id] + word_ids + [wtokenizer.sep_token_id], [pad] + tokens + [pad]


def measure_ner_accuracy(predictions: List[List[int]], truths: List[List[int]], pad: int, mapping: Dict[int, str], \
        offset: int) -> Tuple[float, float, float]:
    def remove_pads(_prediction: List[int], _truth: List[int]) -> Tuple[List[int], List[int]]:
        _prediction = _prediction[:len(_truth)]
        return [_p for i, _p in enumerate(_prediction) if _truth[i] != pad], [_t for _t in _truth if _t != pad]

    def convert_to_str(_prediction: List[int]) -> List[str]:
        return [mapping[_p + offset] for _p in _prediction]

    pairs = tuple(map(remove_pads, predictions, truths))
    predictions, truths = [pair[0] for pair in pairs], [pair[1] for pair in pairs]
    predictions_str: List[str] = reduce(add, list(map(convert_to_str, predictions)))
    truths_str: List[str] = reduce(add, list(map(convert_to_str, truths)))
    return evaluate(truths_str, predictions_str, True)


def main(sonar_path: str, device: str, batch_size_train: int, batch_size_dev: int,
         num_epochs: int) -> None:
    def sprint(s: str) -> None:
        print(s)
        sys.stdout.flush()

    tokenizer = RobertaTokenizer.from_pretrained("pdelobelle/robbert-v2-dutch-base")
    word_pad_id = tokenizer.pad_token_id
    token_pad_id = -5
    offset = 0
    loss_fn = CrossEntropyLoss(ignore_index=token_pad_id, reduction='mean')
    

    sonar_ner = create_sonar_ner(sonar_path)
    class_map = sonar_ner.class_map
    processed_train = tokenize_data(tokenizer, [t for t in sonar_ner.train_data if len(t) <= 100], token_pad_id, offset)
    processed_dev = tokenize_data(tokenizer, [t for t in sonar_ner.dev_data if len(t) <= 100], token_pad_id, offset)
    processed_test = tokenize_data(tokenizer, [t for t in sonar_ner.test_data if len(t) <= 100], token_pad_id, offset)

    train_loader = DataLoader(dataset=TokenDataset(processed_train), batch_size=batch_size_train, shuffle=True,
                              collate_fn=token_collator(word_pad_id, token_pad_id))
    dev_loader = DataLoader(dataset=TokenDataset(processed_dev), batch_size=batch_size_dev, shuffle=False,
                            collate_fn=token_collator(word_pad_id, token_pad_id))
    test_loader = DataLoader(dataset=TokenDataset(processed_test), batch_size=batch_size_dev, shuffle=False,
                             collate_fn=token_collator(word_pad_id, token_pad_id))

    model = RobertaForTokenClassification.from_pretrained("pdelobelle/robbert-v2-dutch-base", num_labels=len(class_map)).to(device)
    optim = AdamW(model.parameters(), lr=3e-05)

    val_truth = [sample[1] for sample in processed_dev]
    test_truth = [sample[1] for sample in processed_test]

    for epoch in range(num_epochs):
        sprint(f'\tEPOCH {epoch+1}:')
        train_loss, train_accu = train_epoch(model, loss_fn, optim, train_loader, token_pad_id, word_pad_id, device)
        sprint(f'Train loss:\t\t{train_loss}')
        sprint(f'Train accu:\t\t{train_accu}')
        sprint('')
        val_loss, val_accu, predictions = eval_epoch(model, loss_fn, dev_loader, token_pad_id, word_pad_id, device)
        val_predictions = measure_ner_accuracy(predictions, val_truth, token_pad_id, class_map, offset)
        sprint(f'Dev loss:\t\t{val_loss}')
        sprint(f'Dev accu:\t\t{val_accu}')
        sprint(f'Scores:\t\t{val_predictions}')
        sprint('')
        test_loss, test_accu, predictions = eval_epoch(model, loss_fn, test_loader, token_pad_id, word_pad_id, device)
        test_predictions = measure_ner_accuracy(predictions, test_truth, token_pad_id, class_map, offset)
        sprint(f'Test loss:\t\t{test_loss}')
        sprint(f'Test accu:\t\t{test_accu}')
        sprint(f'Scores:\t\t{test_predictions}')
        sprint('-' * 120)

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-p', '--sonar_path', help='Path to sonar ner folder.')
    parser.add_argument('-d', '--device', help='Which device to use', default='cuda')
    parser.add_argument('-b', '--batch_size_train', help='Training batch size', default=32, type=int)
    parser.add_argument('-bd', '--batch_size_dev', help='Validation batch size', default=512, type=int)
    parser.add_argument('-e', '--num_epochs', help='How many epochs to train for', default=10, type=int)

    kwargs = vars(parser.parse_args())

    main(**kwargs)
