from TypeLM.preprocessing.defaults import default_tokenizer
from nlp_nl.nl_eval.datasets import create_ner
from TypeLM.finetuning.token_level import (tokenize_data, TokenDataset, token_collator, DataLoader, CrossEntropyLoss,
                                           TypedLMForTokenClassification, default_pretrained, train_epoch, eval_epoch)
from torch.optim import AdamW

tokenizer = default_tokenizer()
word_pad_id = tokenizer.word_tokenizer.core.pad_token_id
token_pad_id = -100
train_batch_size = 32
val_batch_size = 512
NER = create_ner('./nlp_nl/NER/')
pretrained_path: str = 'placeholder'
device: str = 'cuda'
num_epochs = 10
loss_fn = CrossEntropyLoss(ignore_index=token_pad_id, reduction='mean')

processed_train = tokenize_data(tokenizer, NER.train_data, token_pad_id)
processed_dev = tokenize_data(tokenizer, NER.dev_data, token_pad_id)
processed_test = tokenize_data(tokenizer, NER.test_data, token_pad_id)

train_loader = DataLoader(dataset=TokenDataset(processed_train), batch_size=train_batch_size, shuffle=True,
                          collate_fn=token_collator(word_pad_id, token_pad_id))
dev_loader = DataLoader(dataset=TokenDataset(processed_train), batch_size=train_batch_size, shuffle=True,
                        collate_fn=token_collator(word_pad_id, token_pad_id))
test_loader = DataLoader(dataset=TokenDataset(processed_train), batch_size=train_batch_size, shuffle=True,
                         collate_fn=token_collator(word_pad_id, token_pad_id))

model = TypedLMForTokenClassification(default_pretrained(pretrained_path), len(NER.class_map)).to(device)
optim = AdamW(model.parameters(), lr=5e-05)

for epoch in range(num_epochs):
    train_loss, train_accu = train_epoch(model, loss_fn, optim, train_loader, token_pad_id, word_pad_id, device)
    print(f'Train loss:\t\t{train_loss}')
    print(f'Train accu:\t\t{train_accu}')
    val_loss, val_accu = eval_epoch(model, loss_fn, dev_loader, token_pad_id, word_pad_id, device)
    print(f'Dev loss:\t\t{val_loss}')
    print(f'Dev accu:\t\t{val_accu}')
