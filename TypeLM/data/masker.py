from typing import Set, Tuple, Callable
from random import random, choice

from TypeLM.data.tokenizer import default_tokenizer, Indexer
from TypeLM.utils.imports import ints
from TypeLM.utils.token_definitions import MASK, PAD


def mask_indices(seq: ints, random_chance: float) -> ints:
    return [1 if random() < random_chance else 0 for _ in range(len(seq))]


def mask_sampling(seq: ints, masked_indices: ints, sampling_strategy: Callable[[int], int]) -> ints:
    return [sampling_strategy(t) if masked_indices[i] == 1 else t for i, t in enumerate(seq)]


def random_replace(x: int, mask_token: int, mask_chance: float, keep_chance: float, replacement: ints) -> int:
    return mask_token if random() < mask_chance else x if random() < keep_chance else choice(replacement)


def unmask_indices(seq: ints, masks: ints, unpredictable: Set[int]) -> ints:
    return [m if seq[i] not in unpredictable else 0 for i, m in enumerate(masks)]


class RandomReplacer(object):
    def __init__(self, mask_token: int, mask_chance: float, keep_chance: float, replacements: ints):
        self.mask_token = mask_token
        self.mask_chance = mask_chance
        self.keep_chance = keep_chance
        self.replacements = replacements  # todo. make dynamic?

    def __call__(self, x: int) -> int:
        return random_replace(x, self.mask_token, self.mask_chance, self.keep_chance, self.replacements)


class Masker(object):
    def __init__(self, outer_chance: float, mask_token: int, inner_chance: float, keep_chance: float,
                 replacements: ints, unpredictable: Set[int]):
        """
            Masker object that handles the masking and replacement functions.

        :param outer_chance:    Chance that a word is hidden from the input.
        :param mask_token:      Token to replace masked words with.
        :param inner_chance:    Chance that a masked word actually gets replaced.
        :param keep_chance:     Chance that a masked word that did not get replaced persists as is.
        :param replacements:    A set of words that can be used to randomly replace a masked word.
        :param unpredictable:   A set of words that incur no loss, even when masked.
        """

        self.mask_chance = outer_chance
        self.random_replacer = RandomReplacer(mask_token, inner_chance, keep_chance, replacements)
        self.unpredictable = unpredictable

    def __call__(self, words: ints) -> Tuple[ints, ints]:
        indices = mask_indices(words, self.mask_chance)
        masked = mask_sampling(words, indices, self.random_replacer)
        return masked, unmask_indices(words, indices, self.unpredictable)


def default_masker() -> Masker:
    tokenizer = default_tokenizer()
    indexer = Indexer(tokenizer)
    return Masker(outer_chance=0.15, mask_token=indexer.index_word(MASK), inner_chance=0.9, keep_chance=0.5,
                  replacements=list(set(indexer.word_indices.values()) - indexer.masked_words),
                  unpredictable=indexer.masked_words)


def non_masker() -> Callable[[ints], Tuple[ints, ints]]:
    return lambda seq: (seq, [0 for _ in range(len(seq))])


def type_masker() -> Masker:
    tokenizer = default_tokenizer()
    indexer = Indexer(tokenizer)
    return Masker(outer_chance=1., mask_token=0, inner_chance=0., keep_chance=0.85,
                  replacements=list(set(indexer.type_indices.values()) - {indexer.index_type(PAD)}),
                  unpredictable={indexer.index_type(PAD)})
