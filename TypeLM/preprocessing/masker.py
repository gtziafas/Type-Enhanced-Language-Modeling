from typing import Set, TypeVar, Generic, Sequence, Callable, Tuple
from random import choice, random

Token = TypeVar('Token')


def mask_token(x: Token, mask: Token, mask_chance: float, keep_chance: float, replacements: Sequence[Token]) -> Token:
    return mask if random() < mask_chance else x if random() < keep_chance else choice(replacements)


def mask_sequence(seq: Sequence[Token], indices: Sequence[int], masker: Callable[[Token], Token]) -> Sequence[Token]:
    return [masker(t) if indices[i] == 1 else t for i, t in enumerate(seq)]


def mask_indices(seq: Sequence[Token], random_chance: float, unmaskable: Set[Token]) -> Sequence[int]:
    return [0 if t in unmaskable or random() > random_chance else 1 for t in seq]


class RandomReplacer(Generic[Token]):
    def __init__(self, token_mask: Token, mask_chance: float, keep_chance: float, replacements: Set[Token]):
        self.token_mask = token_mask
        self.mask_chance = mask_chance
        self.keep_chance = keep_chance
        self.replacements = tuple(replacements)

    def __call__(self, x: Token) -> Token:
        return mask_token(x, self.token_mask, self.mask_chance, self.keep_chance, self.replacements)


class Masker(Generic[Token]):
    def __init__(self, outer_chance: float, mask_chance: float, keep_chance: float, replacements: Set[Token],
                 token_mask: Token, unmaskable: Set[Token]):
        """
            Masker class that handles masking and replacement functions.

        :param outer_chance:    Chance that a token is hidden from the input.
        :param mask_chance:     Chance that a hidden token is replaced with the [MASK] token.
        :param keep_chance:     Chance that a hidden token is retained.
        :param replacements:    A set of tokens that can be used as replacements.
        :param token_mask:      The [MASK] token.
        :param unmaskable:      A set of tokens that can never be masked.
        """
        self.outer_chance = outer_chance
        self.unmaskable = unmaskable
        self.replacer = RandomReplacer(token_mask, mask_chance, keep_chance, replacements)

    def __call__(self, sent: Sequence[Token]) -> Tuple[Sequence[int], Sequence[Token]]:
        indices = mask_indices(sent, self.outer_chance, self.unmaskable)
        return indices, mask_sequence(sent, indices, self.replacer.__call__)