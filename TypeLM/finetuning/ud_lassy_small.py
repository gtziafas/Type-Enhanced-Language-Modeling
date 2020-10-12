from TypeLM.preprocessing.defaults import default_tokenizer
from nlp_nl.nl_eval.datasets import create_ud_lassy_small
from TypeLM.finetuning.token_level import (tokenize_data, TokenDataset, DataLoader, CrossEntropyLoss, tensor,
                                           TypedLMForTokenClassification, default_pretrained, token_collator,
                                           train_epoch, eval_epoch, train_batch, eval_batch)
from torch.optim import AdamW
from typing import List, Dict, Tuple, Callable
import sys


def main(udls_path: str, model_path: str, device: str, batch_size_train: int, batch_size_dev: int,
         num_epochs: int) -> None:
    def sprint(s: str) -> None:
        print(s)
        sys.stdout.flush()

    tokenizer = default_tokenizer()
    word_pad_id = tokenizer.word_tokenizer.core.pad_token_id
    token_pad_id = -100
    offset = 0
    loss_fn = CrossEntropyLoss(ignore_index=token_pad_id, reduction='mean')

    udls = create_ud_lassy_small(udls_path)
    processed_train = tokenize_data(tokenizer, [t for t in udls.train_data if len(t) <= 100], \
        token_pad_id, offset)
    processed_dev = tokenize_data(tokenizer, [t for t in udls.dev_data if len(t) <= 100], \
        token_pad_id, offset)
    processed_test = tokenize_data(tokenizer, [t for t in udls.test_data if len(t) <= 100], \
        token_pad_id, offset)

    train_loader = DataLoader(dataset=TokenDataset(processed_train), batch_size=batch_size_train, shuffle=True,
                              collate_fn=token_collator(word_pad_id, token_pad_id))
    dev_loader = DataLoader(dataset=TokenDataset(processed_dev), batch_size=batch_size_dev, shuffle=False,
                            collate_fn=token_collator(word_pad_id, token_pad_id))
    test_loader = DataLoader(dataset=TokenDataset(processed_test), batch_size=batch_size_dev, shuffle=False,
                             collate_fn=token_collator(word_pad_id, token_pad_id))

    model = TypedLMForTokenClassification(default_pretrained(model_path), len(udls.class_map)).to(device)
    optim = AdamW(model.parameters(), lr=3e-05)

    sprint('Done with tokenization/loading, starting to train...')
    for epoch in range(num_epochs):
        sprint(f'\tEPOCH {epoch+1}:')
        train_loss, train_accu = train_epoch(model, loss_fn, optim, train_loader, token_pad_id, word_pad_id, device)
        sprint(f'Train loss:\t\t{train_loss:.5f}')
        sprint(f'Train accu:\t\t{train_accu:.5f}')
        sprint('')
        val_loss, val_accu = eval_epoch(model, loss_fn, dev_loader, token_pad_id, word_pad_id, device)
        sprint(f'Dev loss:\t\t{val_loss:.5f}')
        sprint(f'Dev accu:\t\t{val_accu:.5f}')
        sprint('')
        test_loss, test_accu = eval_epoch(model, loss_fn, test_loader, token_pad_id, word_pad_id, device)
        sprint(f'Test loss:\t\t{test_loss:.5f}')
        sprint(f'Test accu:\t\t{test_accu:.5f}')
        sprint('-' * 64)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-p', '--udls_path', help='Path to UD Lassy-Small folder.')
    parser.add_argument('-m', '--model_path', help='Path to pretrained model')
    parser.add_argument('-d', '--device', help='Which device to use', default='cuda')
    parser.add_argument('-b', '--batch_size_train', help='Training batch size', default=32, type=int)
    parser.add_argument('-bd', '--batch_size_dev', help='Validation batch size', default=512, type=int)
    parser.add_argument('-e', '--num_epochs', help='How many epochs to train for', default=10, type=int)

    kwargs = vars(parser.parse_args())

    main(**kwargs)
