from TypeLM.utils.imports import * 
from TypeLM.utils.utils import load_model, ElementWiseFusion, one_hot_embedding
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
    num_layers = 8
    num_heads = 8
    device = 'cuda'

    encoder_params = {'module_maker': EncoderLayer,
                      'num_layers': num_layers,
                      'num_heads': num_heads,
                      'd_model': d_model,
                      'd_ff': d_ff,
                      'd_k': d_k,
                      'd_v': d_v,
                      'activation_fn': F.gelu}
    type_pred_params = {'in_features': d_model, 'out_features': num_types}
    label_smoother_params = {'smoothing': 0.1, 'num_classes': num_types}

    model =  TypeFactoredLM(masked_encoder=Encoder,
                          type_classifier=Linear,
                          num_words=num_words,
                          masked_encoder_kwargs=encoder_params,
                          type_classifier_kwargs=type_pred_params,
                          fusion=ElementWiseFusion,
                          fusion_kwargs={'activation': torch.tanh},
                          type_embedder=Linear,
                          type_embedder_kwargs={'in_features': num_types, 'out_features': d_model},
                          label_smoother_kwargs=label_smoother_params
                          ).to(device)

    return load_model(model_path=load_id, model=model, opt=torch.optim.Adam(model.parameters()))[0]


def infer_words(sentence: List[int], masked_indices: List[int], model: Module, mask_token: int, 
                kappa: int=10, guidance: Optional[List[str]]=None, confidence: float=0, num_types: int=1186) -> List[int]:
    sentence = torch.tensor(sentence, dtype=torch.long, device=device)
    masked_indices = torch.tensor(masked_indices, dtype=torch.long, device=device)
    sentence[masked_indices==1] = mask_token
    pad_mask = torch.ones(sentence.shape[0], sentence.shape[0], dtype=torch.long, device=device)

    types = None
    if guidance is not None:
        types = - torch.ones(sentence.shape[0], num_types, dtype=torch.float, device=device)
        types[masked_indices==1, :] = one_hot_embedding(torch.tensor(guidance, dtype=torch.long, device=device), num_labels=num_types)

    word_preds = model(sentence.unsqueeze(0), pad_mask, type_guidance=types.unsqueeze(0), confidence=confidence)[0].squeeze(0)
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
        masked_indices = input('Give input mask: ')
        guidance = input('Give type guidance for masked word tokens: ')

        word_indices = indexer.index_sentence(tokenizer.tokenize_sentence(sentence_str, add_eos=True))

        type_preds = infer_types(word_indices, model)
        infered_types = list(map(indexer.inverse_type, type_preds))

        print('Infered types = \n{}'.format('\n'.join(infered_types)))

        if masked_indices is not '':
            word_preds = infer_words(sentence=word_indices, masked_indices=list(map(eval, masked_indices.split(' '))), 
                                     model=model, mask_token=indexer.index_word(MASK), 
                                     guidance=indexer.index_type_sequence(list(map(tokenizer.tokenize_type, guidance.split(' ')))), 
                                     confidence = 0.5)
            infered_words = list(map(indexer.inverse_word, [w for p in word_preds for w in p]))
            print('Infered words = {}'.format('\n'.join(infered_words)))


if __name__ == "__main__":
    main()


