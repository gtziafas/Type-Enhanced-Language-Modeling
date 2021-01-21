from TypeLM.preprocessing.defaults import default_tokenizer
from TypeLM.finetuning.bisequence_level import *
from torch.optim import AdamW
from typing import TypeVar, Tuple
import sys


def proc_sent(sample: Tuple[str, str, str, float]) -> Sample:
    ws1, ws2, label, _ = sample
    return ws1.split(), ws2.split(), 0 if label == 'CONTRADICTION' else 1 if label == 'NEUTRAL' else 2


class SICK(object):
    def __init__(self, sick_fn: str):
        self.sick_fn = sick_fn
        self.name = self.sick_fn.split('/')[-1].split('.')[0]
        self.data = self.load_data()
        self.train_data, self.dev_data, self.test_data = self.split_data()

    def load_data(self):
        with open(self.sick_fn, 'r') as in_file:
            lines = [ln.strip().split('\t') for ln in in_file.readlines()[1:]]
        sentence_data = [tuple(ln[1:5]+ln[-1:]) for ln in lines]
        sentence_data = [(s1, s2, el, float(rl), split)
                         for (s1, s2, el, rl, split) in sentence_data]
        return sentence_data

    def split_data(self) -> Tuple[Samples, Samples, Samples]:
        train_data, dev_data, test_data = [], [], []
        for (s1, s2, el, rl, s) in self.data:
            if s == 'TRAIN':
                train_data.append((s1, s2, el, rl))
            if s == 'TRIAL':
                dev_data.append((s1, s2, el, rl))
            if s == 'TEST':
                test_data.append((s1, s2, el, rl))
        return tuple(map(lambda dset: list(map(proc_sent, dset)), [train_data, dev_data, test_data]))


_T = TypeVar('_T')


def main(sick_path: str, model_path: str, device: str, batch_size_train: int, batch_size_dev: int,
         num_epochs: int) -> None:

    def sprint(s: str) -> None:
        print(s)
        sys.stdout.flush()

    # def subsample(xs: List[_T], maxlen: int) -> List[_T]:
    #     return xs if len(xs) < maxlen else xs[-maxlen:]

    tokenizer = default_tokenizer()
    word_pad_id = tokenizer.word_tokenizer.core.pad_token_id
    loss_fn = CrossEntropyLoss(reduction='mean')

    sick = SICK(sick_path)
    train, dev, test = tuple(map(lambda data: BiSequenceDataset(tokenize_data(tokenizer, data)),
                                 [sick.train_data, sick.dev_data, sick.test_data]))

    train_loader = DataLoader(dataset=train, batch_size=batch_size_train // 2, shuffle=True,
                              collate_fn=sequence_collator(word_pad_id))
    dev_loader = DataLoader(dataset=dev, batch_size=batch_size_dev // 2, shuffle=False,
                            collate_fn=sequence_collator(word_pad_id))
    test_loader = DataLoader(dataset=test, batch_size=batch_size_dev // 2, shuffle=False,
                             collate_fn=sequence_collator(word_pad_id))

    model = TypedLMForBiSequenceClassification(default_pretrained(model_path), 3).to(device)
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
    parser.add_argument('-p', '--sick_path', help='Path to SICK file.')
    parser.add_argument('-m', '--model_path', help='Path to pretrained model')
    parser.add_argument('-d', '--device', help='Which device to use', default='cuda')
    parser.add_argument('-b', '--batch_size_train', help='Training batch size', default=32, type=int)
    parser.add_argument('-bd', '--batch_size_dev', help='Validation batch size', default=512, type=int)
    parser.add_argument('-e', '--num_epochs', help='How many epochs to train for', default=10, type=int)

    kwargs = vars(parser.parse_args())
    main(**kwargs)
