from collections import Counter
from typing import Sequence, Iterable, Dict, Tuple, Callable, List

from functools import reduce 
from operator import add  

from collections import defaultdict

from itertools import chain

from string import ascii_letters, digits
import unicodedata

from TypeLM.utils.token_definitions import NUM, PROC, UNK, MWU

_keep = ascii_letters + digits


def merge_dicts(dicts: Iterable[Dict[str, int]]) -> Dict[str, int]:
    return reduce(add, dicts, dict())


def word_preprocess(word: str) -> List[str]:
    def normalize_accents(word_: str) -> str:
        return unicodedata.normalize('NFKD', word_)

    def replace_nums(word_: str) -> str:
        return NUM if word_.isnumeric() else word_

    def remove_weird_chars(word_: str) -> str:
        return ''.join((c for c in word_ if c in _keep))

    def rename_empty(word_: str) -> str:
        return word_ if len(word_) else PROC

    norm = normalize_accents(word.lower())
    splits = norm.split()
    return list(map(lambda subword:
                    rename_empty(replace_nums(remove_weird_chars(subword))),
                    splits))


def type_preprocess(type_: str) -> List[str]:
    # todo.
    return [type_]


def pair_preprocess(pair: Tuple[str, str]) -> Sequence[Tuple[str, str]]:
    """
        A pair is a tuple (word, type)
    """
    words = word_preprocess(pair[0])
    types = [MWU if i > 0 else type_preprocess(pair[1]) for i in range(len(words))]
    return list(zip(words, types))


def sentence_preprocess(sentence: Sequence[Tuple[str, str]]) -> Sequence[Tuple[str, str]]:
    """
        A sentence is a sequence of (word, type) tuples.
    """
    return list(chain.from_iterable(list(map(pair_preprocess, sentence))))


def get_vocabs_one_file(file: str) -> Tuple[Dict[str, int], Dict[str, int]]:
    with open(file, 'r') as fp:
        lines = fp.readlines()

    words = filter(lambda line: line.startswith('\t\t<word>"'), lines)
    words = map(lambda line: (line.split('<word>"')[1]).split('"</word>')[0], words)

    types = filter(lambda line: line.startswith('\t\t<type>"'), lines)
    types = map(lambda line: (line.split('<type>"')[1]).split('"</type>')[0], types)

    words = Counter(words)
    types = Counter(types)

    return words, types


def get_vocabs_one_thread(files: Sequence[str], start: int, end: int) -> Tuple[Dict[str, int], Dict[str, int]]:
    dicts = map(get_vocabs_one_file, files[start:end])
    words, types = map(merge_dicts, zip(*dicts))
    return words, types


def normalize_vocab(counter: Dict[str, int], normalize_fn: Callable[[str], List[str]]) -> Dict[str, int]:
    new_dict = defaultdict(lambda: 0)
    for key_ in counter.keys():
        subkeys = normalize_fn(key_)
        for subkey in subkeys:
            new_dict[subkey] = new_dict[subkey] + counter[key_]
    return new_dict


def normalize_word_vocab(word_vocab: Dict[str, int]) -> Dict[str, int]:
    return normalize_vocab(word_vocab, word_preprocess)


def normalize_type_vocab(type_vocab: Dict[str, int]) -> Dict[str, int]:
    return normalize_vocab(type_vocab, type_preprocess)


def threshold(counter: Dict[str, int], cutoff: int) -> Dict[str, int]:
    return {k: v for k, v in filter(lambda pair: pair[1] > cutoff, counter.items())}


def map_to_idx(counter: Dict[str, int], defaults: Dict[str, int]) -> Dict[str, int]:
    keys = sorted(((k, v) for k, v in counter.items()), key=lambda x: x[1], reverse=True)
    keys = map(lambda x: x[0], keys)
    keys = {**defaults, **{k: i + len(defaults) for i, k in enumerate(keys)}}
    return defaultdict(lambda: keys[UNK], keys)
