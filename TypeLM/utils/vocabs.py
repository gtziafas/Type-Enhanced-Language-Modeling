from collections import Counter
from typing import Sequence, Iterable, Dict, Tuple, Callable, List

from functools import reduce 
from operator import add  

from collections import defaultdict


def merge_dicts(dicts: Iterable[Dict[str, int]]) -> Dict[str, int]:
    return reduce(add, dicts, dict())


def word_preprocess(word: str) -> List[str]:
    # todo.
    return word.lower().split()


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
    new_dict = defaultdict(lambda : 0)
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
