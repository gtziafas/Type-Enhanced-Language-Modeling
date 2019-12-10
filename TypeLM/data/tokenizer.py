from typing import Set, List
from itertools import product
from TypeLM.utils.token_definitions import UNK, PAD
from TypeLM.data.vocab import word_preprocess, sentence_preprocess, Pairs
import pickle


class Tokenizer(object):
    def __init__(self, vocab: Set[str], prefixes: Set[str], suffixes: Set[str],
                 tokens: Set[str], types: Set[str]):
        self.vocab = vocab
        self.prefixes = sorted(prefixes, key=len, reverse=True)
        self.suffixes = sorted(suffixes, key=len, reverse=True)
        self.tokens = tokens
        self.wraps = sorted(product(prefixes, suffixes),
                            key=lambda pair: (len(pair[0]) + len(pair[1]), len(pair[0])),
                            reverse=True)
        self.types = types

    def __call__(self, sentence: str) -> List[str]:
        return self.tokenize_sentence(sentence)

    def tokenize_word(self, word: str):
        return word if word in self.vocab.union(self.tokens) else self.wrap(word)

    def tokenize_type(self, type_: str):
        return type_ if type_ in self.types else PAD

    def tokenize_sentence(self, sentence: str) -> List[str]:
        preprocessed = word_preprocess(sentence)
        return list(map(self.tokenize_word, preprocessed))

    def tokenize_typed_sentence(self, sentence: Pairs) -> Pairs:
        preprocessed = sentence_preprocess(sentence)
        return list(map(lambda pair:
                        (self.tokenize_word(pair[0]),
                         self.tokenize_type(pair[1])),
                        preprocessed))

    def wrap(self, word: str) -> str:
        wraps = filter(lambda wrap: sum(map(len, wrap)) < len(word), self.wraps)
        for prefix, suffix in wraps:
            if word.startswith(prefix) and word.endswith(suffix):
                return prefix + '##' + suffix
        suffixes = filter(lambda suffix: len(suffix) < len(word), self.suffixes)
        for suffix in suffixes:
            if word.endswith(suffix):
                return '##' + suffix
        prefixes = filter(lambda prefix: len(prefix) < len(word), self.prefixes)
        for prefix in prefixes:
            if word.startswith(prefix):
                return prefix + '##'
        return UNK


def default_tokenizer():
    with open('./TypeLM/data/tokenizer_data.p', 'rb') as f:
        top, prefixes, suffixes, _ = pickle.load(f)
    return Tokenizer(top, prefixes, suffixes, lexical_input_tokens)
