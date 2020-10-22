from TypeLM.preprocessing.defaults import default_tokenizer
from nlp_nl.nl_eval.datasets import create_diedat
from TypeLM.finetuning.token_level import (tokenize_data, TokenDataset, DataLoader, CrossEntropyLoss, tensor,
                                           TypedLMForTokenClassification, default_pretrained, token_collator,
                                           Samples, Tensor, LongTensor, Module, train_batch, eval_batch, TypedLM)
from torch.optim import AdamW, Optimizer
from torch import long, bool, zeros_like, where, logical_or, ones_like, no_grad
from typing import List, Dict, Tuple, Callable
import pickle
import sys
import os


_PROC_FILE = 'proc_filtered.p'


def sprint(s: str) -> None:
    print(s)
    sys.stdout.flush()


def train_epoch(model: TypedLMForTokenClassification, loss_fn: Module, optim: Optimizer,
                dataloader: DataLoader, token_pad: int, word_pad: int, mask_token: int,
                device: str) -> Tuple[float, float]:
    epoch_loss, sum_tokens, sum_correct_tokens = 0., 0, 0

    for words, tokens in dataloader:
        padding_mask = (words != word_pad).unsqueeze(1).repeat(1, words.shape[1], 1).long().to(device)
        words = words.to(device)
        tokens = tokens.to(device)

        # masking die/dat word ids and ignoring all other tokens for loss + accuracy computation
        mask = tokens > 0
        words[mask] = mask_token
        tokens -= 1
        tokens[mask==0] = token_pad

        loss, (batch_total, batch_correct) = train_batch(model, loss_fn, optim, words, \
            padding_mask, tokens, token_pad)
        sum_tokens += batch_total
        sum_correct_tokens += batch_correct
        epoch_loss += loss
    return epoch_loss / len(dataloader), sum_correct_tokens / sum_tokens


def eval_epoch(model: TypedLMForTokenClassification, loss_fn: Module, dataloader: DataLoader, token_pad: int,
               word_pad: int, mask_token: int, device: str) -> Tuple[float, float]:
    epoch_loss, sum_tokens, sum_correct_tokens = 0., 0, 0

    for words, tokens in dataloader:
        padding_mask = (words != word_pad).unsqueeze(1).repeat(1, words.shape[1], 1).long().to(device)
        words = words.to(device)
        tokens = tokens.to(device)

        # masking die/dat word ids and ignoring all other tokens for loss + accuracy computation
        mask = zeros_like(tokens, dtype=bool, device=device)
        mask = mask.masked_fill_(tokens>0, 1)
        words[mask] = mask_token
        tokens -= 1
        tokens[mask==0] = token_pad

        loss, (batch_total, batch_correct), preds = eval_batch(model, loss_fn, words, \
            padding_mask, tokens, token_pad)
        sum_tokens += batch_total
        sum_correct_tokens += batch_correct
        epoch_loss += loss
    return epoch_loss / len(dataloader), sum_correct_tokens / sum_tokens


def main(diedat_path: str, model_path: str, device: str, batch_size_train: int, batch_size_dev: int,
         num_epochs: int, zero_shot: bool, checkpoint: bool) -> None:

    tokenizer = default_tokenizer()
    mask_token_id = tokenizer.word_tokenizer.core.mask_token_id
    word_pad_id = tokenizer.word_tokenizer.core.pad_token_id
    token_pad_id = -100
    loss_fn = CrossEntropyLoss(ignore_index=token_pad_id, reduction='mean')

    if not checkpoint:
        diedat = create_diedat(diedat_path)
        class_map = diedat.class_map
        processed_train = tokenize_data(tokenizer, [t for t in diedat.train_data if len(t) <= 100], \
            token_pad_id)
        processed_dev = tokenize_data(tokenizer, [t for t in diedat.dev_data if len(t) <= 100], \
            token_pad_id)
        processed_test = tokenize_data(tokenizer, [t for t in diedat.test_data if len(t) <= 100], \
            token_pad_id)
        pickle.dump((class_map,
                     processed_train, 
                     processed_dev,
                     processed_test),
                    open(os.path.join(diedat_path, _PROC_FILE), 'wb'))
    else:
        class_map, processed_train, processed_dev, processed_test = pickle.load(
            open(os.path.join(diedat_path, _PROC_FILE), 'rb'))           

    train_loader = DataLoader(dataset=TokenDataset(processed_train), batch_size=batch_size_train, shuffle=True,
                              collate_fn=token_collator(word_pad_id, token_pad_id))
    dev_loader = DataLoader(dataset=TokenDataset(processed_dev), batch_size=batch_size_dev, shuffle=False,
                            collate_fn=token_collator(word_pad_id, token_pad_id))
    test_loader = DataLoader(dataset=TokenDataset(processed_test), batch_size=batch_size_dev, shuffle=False,
                             collate_fn=token_collator(word_pad_id, token_pad_id))

    model = TypedLMForTokenClassification(default_pretrained(model_path), 2).to(device)
    optim = AdamW(model.parameters(), lr=3e-05)

    sprint('Done with tokenization/loading, starting to train...')
    for epoch in range(num_epochs):
        sprint(f'\tEPOCH {epoch+1}:')
        train_loss, train_accu = train_epoch(model, loss_fn, optim, train_loader, \
                token_pad_id, word_pad_id, mask_token_id, device)
        sprint(f'Train loss:\t\t{train_loss:.5f}')
        sprint(f'Train accu:\t\t{train_accu:.5f}')
        sprint('')
        val_loss, val_accu = eval_epoch(model, loss_fn, dev_loader, token_pad_id, \
                word_pad_id, mask_token_id, device)
        sprint(f'Dev loss:\t\t{val_loss:.5f}')
        sprint(f'Dev accu:\t\t{val_accu:.5f}')
        sprint('')
        test_loss, test_accu = eval_epoch(model, loss_fn, test_loader, token_pad_id, \
                word_pad_id, mask_token_id, device)
        sprint(f'Test loss:\t\t{test_loss:.5f}')
        sprint(f'Test accu:\t\t{test_accu:.5f}')
        sprint('-' * 64)


@no_grad()
def zero_shot_eval(model: TypedLM, dataloader: DataLoader, token_pad: int,
                   word_pad: int, mask_token: int, device: str) -> Tuple[int, int]:
    model.eval()

    die_tokens = [v for k, v in model.tokenizer.word_tokenizer.core.vocab.items() if k in {'die', 'Die'}]
    dat_tokens = [v for k, v in model.tokenizer.word_tokenizer.core.vocab.items() if k in {'dat', 'Dat'}]

    sum_tokens, sum_correct_tokens = 0, 0

    for words, tokens in dataloader:
        padding_mask = (words != word_pad).unsqueeze(1).repeat(1, words.shape[1], 1).long().to(device)
        words = words.to(device)
        tokens = tokens.to(device)

        # masking die/dat word ids and ignoring all other tokens for loss + accuracy computation
        mask = zeros_like(tokens, dtype=bool, device=device)
        mask = mask.masked_fill_(tokens>0, 1)
        words[mask] = mask_token
        tokens -= 1
        tokens[mask==0] = token_pad

        contextualized = model.encode(words, padding_mask)[1][mask]
        contextualized = model.word_embedder.invert(contextualized)
        die_prob = contextualized[..., die_tokens[0]] + contextualized[..., die_tokens[1]]
        dat_prob = contextualized[..., dat_tokens[0]] + contextualized[..., dat_tokens[1]]
        predictions = where(die_prob > dat_prob, zeros_like(die_prob), ones_like(dat_prob))

        sum_tokens += predictions.shape[0]
        sum_correct_tokens += (predictions == tokens[mask]).sum().item()
    return sum_correct_tokens, sum_tokens


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-p', '--diedat_path', help='Path to diedat folder.')
    parser.add_argument('-m', '--model_path', help='Path to pretrained model')
    parser.add_argument('-d', '--device', help='Which device to use', default='cuda')
    parser.add_argument('-b', '--batch_size_train', help='Training batch size', default=32, type=int)
    parser.add_argument('-bd', '--batch_size_dev', help='Validation batch size', default=512, type=int)
    parser.add_argument('-e', '--num_epochs', help='How many epochs to train for', default=10, type=int)
    parser.add_argument('--zero_shot', dest='zero_shot', action='store_true', help='Whether to go zero-shot')
    parser.add_argument('--checkpoint', dest='checkpoint', action='store_true', default=False,
        help='Whether to load tokenized data from checkpoint or start from scratch')

    kwargs = vars(parser.parse_args())
    if kwargs['zero_shot']:
        model = default_pretrained(kwargs['model_path'])().to(kwargs['device'])
        _, _, _, processed_test = pickle.load(open(os.path.join(kwargs['diedat_path'], _PROC_FILE), "rb"))
        word_pad, token_pad, mask_pad = (model.tokenizer.word_tokenizer.core.pad_token_id,
                                         -100,
                                         model.tokenizer.word_tokenizer.core.mask_token_id)
        test_loader = DataLoader(dataset=TokenDataset(processed_test), batch_size=kwargs['batch_size_dev'],
                                 shuffle=False, collate_fn=token_collator(word_pad, token_pad))
        sprint('Starting zero-shot evaluation.')
        corr, total = zero_shot_eval(model, test_loader, token_pad, word_pad, mask_pad, kwargs['device'])
        sprint(f'Total: {total}\t Correct: {corr}\t (%): {100 * corr/total:.3f}')
    else:
    	main(**kwargs)
