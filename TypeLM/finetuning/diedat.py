from TypeLM.preprocessing.defaults import default_tokenizer
from nlp_nl.nl_eval.datasets import create_diedat
from TypeLM.finetuning.token_level import (tokenize_data, TokenDataset, token_collator, DataLoader, CrossEntropyLoss,
                                           TypedLMForTokenClassification, default_pretrained, train_epoch, eval_epoch)
from torch.optim import AdamW
from typing import List, Dict, Tuple
import sys


def clean_diedat_data(data: List[Tuple[List[str], List[int]]],
            null_char: str = '\xad') -> List[List[Tuple[str, int]]]:
    data = [list(zip(*t)) for t in data]
    for datum in data:
        words, tokens = zip(*datum)
        if null_char in words:
            data.remove(datum)
    return data


def main(diedat_path: str, model_path: str, device: str, batch_size_train: int, batch_size_dev: int,
         num_epochs: int) -> None:
    def sprint(s: str) -> None:
        print(s)
        sys.stdout.flush()

    tokenizer = default_tokenizer()
    word_pad_id = tokenizer.word_tokenizer.core.pad_token_id
    token_pad_id = -100
    diedat = create_diedat(diedat_path)
    offset = 0
    loss_fn = CrossEntropyLoss(ignore_index=token_pad_id, reduction='mean')

    processed_train = tokenize_data(tokenizer, [t for t in clean_diedat_data(diedat.train_data) if len(t) <= 100] \
        token_pad_id, offset)
    processed_dev = tokenize_data(tokenizer, [t for t in clean_diedat_data(diedat.dev_data) if len(t) <= 100], \
        token_pad_id, offset)
    processed_test = tokenize_data(tokenizer, [t for t in clean_diedat_data(diedat.test_data) if len(t) <= 100], \
        token_pad_id, offset)

    train_loader = DataLoader(dataset=TokenDataset(processed_train), batch_size=batch_size_train, shuffle=True,
                              collate_fn=token_collator(word_pad_id, token_pad_id))
    dev_loader = DataLoader(dataset=TokenDataset(processed_dev), batch_size=batch_size_dev, shuffle=False,
                            collate_fn=token_collator(word_pad_id, token_pad_id))
    test_loader = DataLoader(dataset=TokenDataset(processed_test), batch_size=batch_size_dev, shuffle=False,
                             collate_fn=token_collator(word_pad_id, token_pad_id))

    model = TypedLMForTokenClassification(default_pretrained(model_path), len(diedat.class_map)).to(device)
    optim = AdamW(model.parameters(), lr=3e-05)

    val_truth = [sample[1] for sample in processed_dev]
    test_truth = [sample[1] for sample in processed_test]

    for epoch in range(num_epochs):
        train_loss, train_accu = train_epoch(model, loss_fn, optim, train_loader, token_pad_id, word_pad_id, device)
        sprint(f'Train loss:\t\t{train_loss}')
        sprint(f'Train accu:\t\t{train_accu}')
        val_loss, val_accu, predictions = eval_epoch(model, loss_fn, dev_loader, token_pad_id, word_pad_id, device)
        sprint(f'Dev loss:\t\t{val_loss}')
        sprint(f'Dev accu:\t\t{val_accu}')
        #sprint(f'Scores:\t\t{predictions}')
        test_loss, test_accu, predictions = eval_epoch(model, loss_fn, test_loader, token_pad_id, word_pad_id, device)
        sprint(f'Dev loss:\t\t{test_loss}')
        sprint(f'Dev accu:\t\t{test_accu}')
        #sprint(f'Scores:\t\t{predictions}')


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
