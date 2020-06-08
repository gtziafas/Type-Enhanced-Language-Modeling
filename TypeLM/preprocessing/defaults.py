from typing import Tuple, Sequence, Set, Callable
from TypeLM.preprocessing.tokenizer import Tokenizer
from TypeLM.preprocessing.masker import Masker

with open('./TypeLM/data/indexing/atomset.txt', 'r') as f:
    atomset = set(f.read().split('\n'))
_tokenizer = Tokenizer(type_vocabulary=atomset, atomic=True)


def non_masker() -> Callable[[Sequence[int]], Tuple[Sequence[int], Sequence[int]]]:
    def wrapped(seq: Sequence[int]):
        return seq, [0 for _ in range(len(seq))]
    return wrapped


def default_tokenizer() -> Tokenizer:
    return _tokenizer


def default_word_masker() -> Masker:
    unmaskable = set(_tokenizer.word_tokenizer.core.all_special_ids)
    replacements = set(_tokenizer.word_tokenizer.core.vocab.values()).difference(unmaskable)
    mask = _tokenizer.word_tokenizer.core.mask_token_id
    return Masker(outer_chance=0.15, mask_chance=0.8, keep_chance=0.5, replacements=replacements, token_mask=mask,
                  unmaskable=unmaskable)


def regularization_type_masker() -> Masker:
    unmaskable = {_tokenizer.type_tokenizer.PAD_TOKEN_ID, _tokenizer.type_tokenizer.SOS_TOKEN_ID,
                  _tokenizer.type_tokenizer.SEP_TOKEN_ID}
    replacements = set(_tokenizer.type_tokenizer.vocabulary.values()).difference(unmaskable)
    return Masker(outer_chance=0.01, mask_chance=0, keep_chance=0, replacements=replacements, token_mask=None,
                  unmaskable=unmaskable)
