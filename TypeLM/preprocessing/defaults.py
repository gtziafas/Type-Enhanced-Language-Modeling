from typing import Tuple, Sequence, Callable
from TypeLM.preprocessing.tokenizer import Tokenizer
from TypeLM.preprocessing.masker import Masker
from TypeLM.preprocessing.loader import LazyLoader, Samples
from TypeLM.neural.utils import pad_sequence
from torch import Tensor, LongTensor, ones

with open('./TypeLM/data/indexing/small_typeset.txt', 'r') as f:
    typeset = set(f.read().split('\n'))
_tokenizer = Tokenizer(type_vocabulary=typeset, atomic=False)


def non_masker() -> Callable[[Sequence[int]], Tuple[Sequence[int], Sequence[int]]]:
    def wrapped(seq: Sequence[int]):
        return seq, [0 for _ in range(len(seq))]
    return wrapped


def default_tokenizer() -> Tokenizer:
    return _tokenizer


def default_word_masker() -> Masker:
    unmaskable = set(_tokenizer.word_tokenizer.core.all_special_ids)
    replacements = set(_tokenizer.word_tokenizer.core.vocab.values()).difference(unmaskable)
    subwords = set({k: v for k, v in _tokenizer.word_tokenizer.core.get_vocab().items() if k.startswith('##')}.values())
    mask = _tokenizer.word_tokenizer.core.mask_token_id
    return Masker(outer_chance=0.15, mask_chance=0.8, keep_chance=0.5, replacements=replacements, token_mask=mask,
                  unmaskable=unmaskable, subwords=subwords)


def regularization_type_masker() -> Masker:
    unmaskable = {_tokenizer.type_tokenizer.PAD_TOKEN_ID}
    replacements = set(_tokenizer.type_tokenizer.vocabulary.values()).difference(unmaskable)
    return Masker(outer_chance=0.01, mask_chance=0, keep_chance=0, replacements=replacements, token_mask=None,
                  unmaskable=unmaskable, subwords=set())


def default_dataloader(path: str, chunk_size: int, batch_size: int) -> LazyLoader:
    word_masker = default_word_masker()
    type_masker = regularization_type_masker()

    def post_proc(sentences: Samples):
        true_words, true_types = list(zip(*sentences))

        word_lens = list(map(len, true_words))

        masked_ids, masked_words = list(zip(*list(map(word_masker, true_words))))
        _, masked_types = list(zip(*list(map(type_masker, true_types))))
        masked_words = pad_sequence(list(map(LongTensor, masked_words)),
                                    _tokenizer.word_tokenizer.core.pad_token_id)
        true_words = pad_sequence(list(map(LongTensor, true_words)),
                                  _tokenizer.word_tokenizer.core.pad_token_id)
        true_types = pad_sequence(list(map(LongTensor, masked_types)),
                                  _tokenizer.type_tokenizer.PAD_TOKEN_ID)
        masked_ids = pad_sequence(list(map(LongTensor, masked_ids)),
                                  0)
        padding_mask = ones(true_words.shape[0], true_words.shape[1], true_words.shape[1])
        # todo: write in pytorch ops
        for i, l in enumerate(word_lens):
            padding_mask[i, :, l::] = 0
        return masked_words, true_words, true_types, padding_mask, masked_ids
    return LazyLoader(path, chunk_size, batch_size, post_proc)
