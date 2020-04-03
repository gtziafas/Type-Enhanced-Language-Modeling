from TypeLM.utils.imports import * 
from TypeLM.utils.utils import load_model 
from TypeLM.data.tokenizer import default_tokenizer, Indexer 
from TypeLM.model.masked_encoder import EncoderLayer, Encoder
from TypeLM.model.type_factored_lm import TypeFactoredLM
from TypeLM.utils.token_definitions import MASK

import torch 

import random
from typing import Tuple, List, Optional, Dict, Any

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = './TypeLM/checkpoints/8layers_1.pth'


def get_default_model(vocab_stats: Tuple[int, int], load_id: str = model_path) -> TypeFactoredLM:
    num_words, num_types = vocab_stats
    d_model = 512
    d_ff = 1024
    d_k, d_v = d_model, d_model
    type_vocab_size, word_vocab_size = num_types, num_words
    num_layers = 8

    encoder_params = {'module_maker': EncoderLayer,
                      'num_layers': num_layers,
                      'num_heads': 8,
                      'd_model': d_model,
                      'd_ff': d_ff,
                      'd_k': d_k,
                      'd_v': d_v,
                      'activation_fn': F.gelu}
    type_pred_params = {'in_features': d_model, 'out_features': type_vocab_size}

    model = TypeFactoredLM(masked_encoder=Encoder,
                          type_classifier=Linear,
                          num_words=word_vocab_size,
                          masked_encoder_kwargs=encoder_params,
                          type_classifier_kwargs=type_pred_params,
                          ).to(device)

    return load_model(model_path=load_id, model=model, opt=torch.optim.Adam(model.parameters()))[0]


def infer_words(sentence: List[int], masked_indices: List[int], model: Module, mask_token: int, kappa: int=10) -> List[int]:
    sentence = torch.tensor(sentence, dtype=torch.long, device=device)
    masked_indices = torch.tensor(masked_indices, dtype=torch.long, device=device)
    sentence[masked_indices==1] = mask_token
    pad_mask = torch.ones(sentence.shape[0], sentence.shape[0], dtype=torch.long, device=device)
    word_preds = model.forward_lm(sentence.unsqueeze(0), pad_mask).squeeze(0)
    return word_preds[masked_indices==1].topk(kappa)[1].tolist()


def infer_types(sentence: List[int], model: Module) -> List[int]:
    sentence = torch.tensor(sentence, dtype=torch.long, device=device)
    pad_mask = torch.ones(sentence.shape[0], sentence.shape[0], dtype=torch.long, device=device)
    type_preds = model.forward_st(sentence.unsqueeze(0), pad_mask).squeeze(0)
    return type_preds.argmax(dim=-1).tolist()


def main():

    tokenizer = default_tokenizer()
    indexer = Indexer(tokenizer)

    model = get_default_model(vocab_stats=(len(indexer.word_indices) + 1, len(indexer.type_indices))).train()

    while True:
        sentence_str = input('Give input sentence: ')
        masked_indices = input('Give input mask (leave empty for only type inference): ')

        word_indices = indexer.index_sentence(tokenizer.tokenize_sentence(sentence_str, add_eos=True))

        type_preds = infer_types(word_indices, model)
        infered_types = list(map(indexer.inverse_type, type_preds))

        print('Infered types = \n{}'.format('\n'.join(infered_types)))

        if masked_indices is not '':
            word_preds = infer_words(sentence=word_indices, masked_indices=list(map(eval, masked_indices.split(' '))), 
                                     model=model, mask_token=indexer.index_word(tokenizer.tokenize_word(MASK)))
            infered_words = list(map(indexer.inverse_word, [w for p in word_preds for w in p]))
            print('Infered sentence = {}'.format(' '.join(infered_words)))


if __name__ == "__main__":
    main()


