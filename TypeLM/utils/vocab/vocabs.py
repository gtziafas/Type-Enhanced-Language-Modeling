from collections import Counter
from typing import Sequence, Iterable, Dict, Tuple, Callable, List

from functools import reduce 
from operator import add, le, ge

from collections import defaultdict

from string import ascii_letters, digits
import unicodedata
from TypeLM.utils.vocab.tokens import NUM, PROC

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


def threshold(counter: Dict[str, int], cutoff: int, op: Callable[[int, int], bool]=le) -> Dict[str, int]:
    return {k: v for k, v in filter(lambda pair: op(cutoff, pair[1]), counter.items())}


def map_to_idx(counter: Dict[str, int]) -> Dict[str, int]:
    NotImplemented
    pass


def go():
    fs = list(range(100))
    fs = ['x0'+str(idx) if idx<10 else 'x' + str(idx) for idx in fs]
    words, types = get_vocabs_one_thread(fs, 0, len(fs))

    print(len(words))
    print(len(types))

    return words, types


if __name__ == "__main__":
    go()
