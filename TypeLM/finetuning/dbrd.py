from TypeLM.preprocessing.defaults import default_tokenizer
from nlp_nl.nl_eval.datasets import create_dbrd
from TypeLM.finetuning.sequence_level import (tokenize_data, SequenceDataset, DataLoader, CrossEntropyLoss,
                                              default_pretrained, sequence_collator, train_epoch, eval_epoch,
                                              TypedLMForSequenceClassification)
from torch.optim import AdamW
from typing import List, TypeVar
import pickle
import sys
import os


_T = TypeVar('_T')


def main(dbrd_path: str, model_path: str, device: str, batch_size_train: int, batch_size_dev: int,
         num_epochs: int, checkpoint: bool) -> None:
    def sprint(s: str) -> None:
        print(s)
        sys.stdout.flush()

    def subsample(xs: List[_T], maxlen: int) -> List[_T]:
        return xs if len(xs) < maxlen else xs[-maxlen:]

    tokenizer = default_tokenizer()
    word_pad_id = tokenizer.word_tokenizer.core.pad_token_id
    loss_fn = CrossEntropyLoss(reduction='mean')

    if not checkpoint:
        dbrd = create_dbrd(dbrd_path)
        class_map = dbrd.class_map
        split = int(0.9 * len(dbrd.train_data))
        processed_train = tokenize_data(tokenizer, [(subsample(ws, 100), t) for (ws, t) in dbrd.train_data[:split]])
        processed_dev = tokenize_data(tokenizer, sorted(
            [(subsample(ws, 100), t) for (ws, t) in dbrd.train_data[split:]], key=lambda x: len(x[0])))
        processed_test = tokenize_data(tokenizer, sorted(
            [(subsample(ws, 100), t) for (ws, t) in dbrd.test_data], key=lambda x: len(x[0])))
        pickle.dump((class_map,
                     processed_train, 
                     processed_dev,
                     processed_test),
                    open(os.path.join(udls_path, 'proc.p'), 'wb'))
    else:
        class_map, processed_train, processed_dev, processed_test = pickle.load(
            open(os.path.join(dbrd_path, 'proc.p'), 'rb'))        

    train_loader = DataLoader(dataset=SequenceDataset(processed_train), batch_size=batch_size_train, shuffle=True,
                              collate_fn=sequence_collator(word_pad_id))
    dev_loader = DataLoader(dataset=SequenceDataset(processed_dev), batch_size=batch_size_dev, shuffle=False,
                            collate_fn=sequence_collator(word_pad_id))
    test_loader = DataLoader(dataset=SequenceDataset(processed_test), batch_size=batch_size_dev, shuffle=False,
                             collate_fn=sequence_collator(word_pad_id))

    model = TypedLMForSequenceClassification(default_pretrained(model_path), 2).to(device)
    optim = AdamW(model.parameters(), lr=3e-05)

    sprint('Done with tokenization/loading, starting to train...')
    for epoch in range(num_epochs):
        sprint(f'\tEPOCH {epoch+1}:')
        train_loss, train_accu = train_epoch(model, loss_fn, optim, train_loader, word_pad_id, device)
        sprint(f'Train loss:\t\t{train_loss:.5f}')
        sprint(f'Train accu:\t\t{train_accu:.5f}')
        sprint('')
        dev_loss, dev_accu = eval_epoch(model, loss_fn, dev_loader, word_pad_id, device)
        sprint(f'Test loss:\t\t{dev_loss:.5f}')
        sprint(f'Test accu:\t\t{dev_accu:.5f}')
        sprint('')
        test_loss, test_accu = eval_epoch(model, loss_fn, test_loader, word_pad_id, device)
        sprint(f'Test loss:\t\t{test_loss:.5f}')
        sprint(f'Test accu:\t\t{test_accu:.5f}')
        sprint('-' * 64)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-p', '--dbrd_path', help='Path to DBRD folder.')
    parser.add_argument('-m', '--model_path', help='Path to pretrained model')
    parser.add_argument('-d', '--device', help='Which device to use', default='cuda')
    parser.add_argument('-b', '--batch_size_train', help='Training batch size', default=32, type=int)
    parser.add_argument('-bd', '--batch_size_dev', help='Validation batch size', default=512, type=int)
    parser.add_argument('-e', '--num_epochs', help='How many epochs to train for', default=10, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint', action='store_true', default=False,
        help='Whether to load tokenized data from checkpoint or start from scratch')

    kwargs = vars(parser.parse_args())

    main(**kwargs)
