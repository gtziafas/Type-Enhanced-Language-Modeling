from TypeLM.finetuning.token_level import tokenize_data
from TypeLM.preprocessing.defaults import default_tokenizer
from nlp_nl.nl_eval.datasets import create_ner
from torch.utils.data import DataLoader
from TypeLM.finetuning.token_level import TokenDataset, token_collator


tokenizer = default_tokenizer()
word_pad_id = tokenizer.word_tokenizer.core.pad_token_id
token_pad_id = -100
train_batch_size = 32
val_batch_size = 512
NER = create_ner('./nlp_nl/NER/')

processed_train = tokenize_data(tokenizer, NER.train_data, token_pad_id)
processed_dev = tokenize_data(tokenizer, NER.dev_data, token_pad_id)
processed_test = tokenize_data(tokenizer, NER.test_data, token_pad_id)

train_loader = DataLoader(dataset=TokenDataset(processed_train), batch_size=train_batch_size, shuffle=True,
                          collate_fn=token_collator(word_pad_id, token_pad_id))
dev_loader = DataLoader(dataset=TokenDataset(processed_train), batch_size=train_batch_size, shuffle=True,
                        collate_fn=token_collator(word_pad_id, token_pad_id))
test_loader = DataLoader(dataset=TokenDataset(processed_train), batch_size=train_batch_size, shuffle=True,
                         collate_fn=token_collator(word_pad_id, token_pad_id))