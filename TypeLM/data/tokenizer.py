from typing import Set, List, Dict, Iterable, Tuple, TextIO
from itertools import product
from TypeLM.utils.token_definitions import UNK, PAD, EOS, PROC, word_input_tokens, type_input_tokens
from TypeLM.data.vocab import word_preprocess, sentence_preprocess, Pairs
from TypeLM.data.vocab import strs, is_type, is_word, extract_type, extract_word
import pickle


class Tokenizer(object):
    def __init__(self, vocab: Set[str], prefixes: Set[str], suffixes: Set[str], types: Set[str]):
        self.vocab = vocab
        self.prefixes = sorted(prefixes, key=len, reverse=True)
        self.suffixes = sorted(suffixes, key=len, reverse=True)
        self.wraps = sorted(product(prefixes, suffixes),
                            key=lambda pair: (len(pair[0]) + len(pair[1]), len(pair[0])),
                            reverse=True)
        self.types = types

    def __call__(self, sentence: str) -> List[str]:
        return self.tokenize_sentence(sentence)

    def tokenize_word(self, word: str):
        return word if word in self.vocab else self.wrap(word)

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
        top, prefixes, suffixes, types = pickle.load(f)
    return Tokenizer(top.union(word_input_tokens), prefixes, suffixes, types.union(type_input_tokens))


class Indexer(object):
    def __init__(self, tokenizer: Tokenizer):
        words = sorted(tokenizer.vocab)
        wraps = list(map('##'.join, tokenizer.wraps))
        prefixes = list(map(lambda pre: pre + '##', tokenizer.prefixes))
        suffixes = list(map(lambda su: '##' + su, tokenizer.suffixes))
        types = sorted(tokenizer.types)

        self._word_indices = self.make_any_indices(words + wraps + prefixes + suffixes, 1)
        self._type_indices = {PAD: 0, **self.make_any_indices(types, 1)}

        masked_words = wraps + prefixes + suffixes + [EOS, UNK, PROC]
        self._masked_words = {self._word_indices[k] for k in masked_words}

    @staticmethod
    def make_any_indices(vocab: Iterable[str], start_from: int) -> Dict[str, int]:
        return {w: i + start_from for i, w in enumerate(vocab)}

    def index_word(self, word: str) -> int:
        return self._word_indices[word]

    def index_sentence(self, sentence: List[str]) -> List[int]:
        return list(map(self.index_word, sentence))

    def index_type(self, type_: str) -> int:
        return self._type_indices[type_]

    def index_type_sequence(self, type_sequence: List[str]) -> List[int]:
        return list(map(self.index_type, type_sequence))

    def index_typed_sentence(self, sentence: Pairs) -> Tuple[List[int], List[int]]:
        words, types = zip(*sentence)
        return self.index_sentence(words), self.index_type_sequence(types)


def stringify(words: List[int], types: List[int]) -> str:
    return ' '.join(map(str, words)) + '\t' + ' '.join(map(str, types)) + '\n'


def index_corpus(read_from: strs, write_to: str) -> None:

    tokenizer = default_tokenizer()
    indexer = Indexer(tokenizer)

    partial = [], []

    with open(write_to, 'a') as output_wrapper:
        for file in read_from:
            with open(file, 'r') as input_wrapper:
                partial = get_partial_samples(input_wrapper, output_wrapper, partial, tokenizer, indexer)


def get_partial_samples(input_wrapper: TextIO, output_wrapper: TextIO, current: Tuple[strs, strs],
                        tokenizer: Tokenizer, indexer: Indexer) -> Tuple[strs, strs]:
    def is_eos(line_: str):
        return line_.startswith('</sentence>')

    words, types = current

    for line in input_wrapper:
        if is_word(line):
            words.append(extract_word(line))
        elif is_type(line):
            types.append(extract_type(line))
        elif is_eos(line):
            sentence = list(zip(words, types))
            tokenized = tokenizer.tokenize_typed_sentence(sentence)
            indexed = indexer.index_typed_sentence(tokenized)
            output_wrapper.write(stringify(*indexed))
            words, types = [], []

    return words, types
