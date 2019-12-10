from typing import Set, List
from itertools import product
from TypeLM.utils.token_definitions import UNK, lexical_input_tokens
from TypeLM.data.vocab import word_preprocess, sentence_preprocess, Pairs
import pickle


class Tokenizer(object):
    def __init__(self, vocab: Set[str], prefixes: Set[str], suffixes: Set[str], tokens: Set[str]):
        self.vocab = vocab
        self.prefixes = prefixes
        self.suffixes = suffixes
        self.tokens = tokens
        self.wraps = sorted(product(prefixes, suffixes),
                            key=lambda pair: (len(pair[0]) + len(pair[1]), len(pair[0])),
                            reverse=True)

    def __call__(self, sentence: str) -> List[str]:
        return self.tokenize_sentence(sentence)

    def tokenize_word(self, word: str):
        return word if word in self.vocab.union(self.tokens) else self.wrap(word)

    def tokenize_sentence(self, sentence: str) -> List[str]:
        preprocessed = word_preprocess(sentence)
        return list(map(self.tokenize_word, preprocessed))

    def tokenize_typed_sentence(self, sentence: Pairs) -> Pairs:
        preprocessed = sentence_preprocess(sentence)
        return list(map(lambda pair: (self.tokenize_word(pair[0]), pair[1]), preprocessed))

    def wrap(self, word: str) -> str:
        wraps = filter(lambda wrap:
                       len(wrap[0]) < len(word) and len(wrap[1]) < len(word) and sum(map(len, wrap)) < len(word),
                       self.wraps)
        for prefix, suffix in wraps:
            if word.startswith(prefix) and word.endswith(suffix):
                return prefix + '##' + suffix
        prefixes = filter(lambda prefix: len(prefix) < len(word), self.prefixes)
        for prefix in prefixes:
            if word.startswith(prefix):
                return prefix + '##'
        suffixes = filter(lambda suffix: len(prefix) < len(word), self.suffixes)
        for suffix in suffixes:
            if word.endswith(suffix):
                return '##' + suffix
        return UNK


def default_tokenizer():
    with open('./TypeLM/data/tokenizer_data.p', 'rb') as f:
        top, prefixes, suffixes = pickle.load(f)
    return Tokenizer(top, prefixes, suffixes, lexical_input_tokens)
