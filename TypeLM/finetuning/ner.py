from TypeLM.preprocessing.defaults import default_tokenizer
from nlp_nl.nl_eval.datasets import create_ner
from TypeLM.finetuning.token_level import (tokenize_data, TokenDataset, token_collator, DataLoader, CrossEntropyLoss,
                                           TypedLMForTokenClassification, default_pretrained, train_epoch, eval_epoch)
from torch.optim import AdamW


def main(ner_path: str, model_path: str, device: str, batch_size_train: int, batch_size_dev: int,
         num_epochs: int) -> None:
    tokenizer = default_tokenizer()
    word_pad_id = tokenizer.word_tokenizer.core.pad_token_id
    token_pad_id = -100
    ner = create_ner(ner_path)
    offset = 1
    loss_fn = CrossEntropyLoss(ignore_index=token_pad_id, reduction='mean')

    processed_train = tokenize_data(tokenizer, [t for t in ner.train_data if len(t) <= 100], token_pad_id, offset)
    processed_dev = tokenize_data(tokenizer, ner.dev_data, token_pad_id, offset)
    # processed_test = tokenize_data(tokenizer, NER.test_data, token_pad_id, offset)

    train_loader = DataLoader(dataset=TokenDataset(processed_train), batch_size=batch_size_train, shuffle=True,
                              collate_fn=token_collator(word_pad_id, token_pad_id))
    dev_loader = DataLoader(dataset=TokenDataset(processed_dev), batch_size=batch_size_dev, shuffle=True,
                            collate_fn=token_collator(word_pad_id, token_pad_id))
    # test_loader = DataLoader(dataset=TokenDataset(processed_test), batch_size=batch_size_dev, shuffle=True,
    #                          collate_fn=token_collator(word_pad_id, token_pad_id))

    model = TypedLMForTokenClassification(default_pretrained(model_path), len(ner.class_map)).to(device)
    optim = AdamW(model.parameters(), lr=5e-05)

    for epoch in range(num_epochs):
        train_loss, train_accu = train_epoch(model, loss_fn, optim, train_loader, token_pad_id, word_pad_id, device)
        print(f'Train loss:\t\t{train_loss}')
        print(f'Train accu:\t\t{train_accu}')
        val_loss, val_accu = eval_epoch(model, loss_fn, dev_loader, token_pad_id, word_pad_id, device)
        print(f'Dev loss:\t\t{val_loss}')
        print(f'Dev accu:\t\t{val_accu}')


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-n', '--ner_path', help='Path to ner folder.')
    parser.add_argument('-m', '--model_path', help='Path to pretrained model')
    parser.add_argument('-d', '--device', help='Which device to use', default='cuda')
    parser.add_argument('-b', '--batch_size_train', help='Training batch size', default=32, type=int)
    parser.add_argument('-bd', '--batch_size_dev', help='Validation batch size', default=512, type=int)
    parser.add_argument('-e', '--num_epochs', help='How many epochs to train for', default=10, type=int)

    kwargs = vars(parser.parse_args())

    main(**kwargs)
