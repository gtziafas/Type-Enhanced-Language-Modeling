from collections import Counter
from typing import Sequence, Iterable, Dict, Tuple, Callable, List, Iterator

from functools import reduce 
from operator import add, lt, ge

from collections import defaultdict

from itertools import chain

from string import ascii_letters, digits
import unicodedata

from TypeLM.utils.token_definitions import NUM, PROC, UNK, MWU, EOS


_keep = ascii_letters + digits


strs = List[str]
Pair = Tuple[str, str]
Pairs = Sequence[Pair]
Sentences = Sequence[Pairs]


def merge_dicts(dicts: Iterable[Dict[str, int]]) -> Dict[str, int]:
    return reduce(add, dicts, dict())


def word_preprocess(word: str) -> strs:
    if word == EOS:
        return [EOS]

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


def type_preprocess(type_: str) -> strs:
    # todo.
    return [type_]


def pair_preprocess(pair: Pair) -> Pairs:
    """
        A pair is a tuple (word, type)
    """
    words = word_preprocess(pair[0])
    types = [MWU if i > 0 else type_preprocess(pair[1])[0] for i in range(len(words))]
    return list(zip(words, types))


def sentence_preprocess(sentence: Pairs) -> Pairs:
    """
        A sentence is a sequence of (word, type) tuples.
    """
    return list(chain.from_iterable(list(map(pair_preprocess, sentence))))


def is_word(line: str) -> bool:
    return line.startswith('\t\t<word>')


def is_type(line: str) -> bool:
    return line.startswith('\t\t<type>')


def extract_word(line: str) -> str:
    return line.split('<word>"')[1].split('"</word>')[0]


def extract_type(line: str) -> str:
    return line.split('<type>"')[1].split('"</type>')[0]


def get_vocabs_one_file(file: str) -> Tuple[Dict[str, int], Dict[str, int]]:
    with open(file, 'r') as fp:
        lines = fp.readlines()

    words = filter(is_word, lines)
    words = map(extract_word, words)

    types = filter(is_type, lines)
    types = map(extract_type, types)

    words = Counter(words)
    types = Counter(types)

    return words, types


def get_vocabs_one_thread(files: strs, start: int, end: int) -> Tuple[Dict[str, int], Dict[str, int]]:
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


def threshold(counter: Dict[str, int], cutoff: int, op: Callable[[int, int], bool]=lt) -> Dict[str, int]:
    return {k: v for k, v in filter(lambda pair: op(cutoff, pair[1]), counter.items())}


def pile_of_shit(counter: Dict[Sequence[str], int], pref_merges, suf_merges, wrap_merges) -> Dict[Sequence[str], int]:

    # suf_pref_dict : Dict[Tuple[str, str], int]
    suf_pref_dict = defaultdict(lambda : 0)
    suf_dict = defaultdict(lambda: 0)
    pref_dict = defaultdict(lambda : 0)

    for word in counter.keys():
        if len(word) == 1:
            continue

        first_ngram = word[0]
        last_ngram = word[-1]
        suf_pref_dict[(first_ngram, last_ngram)] = suf_pref_dict[(first_ngram, last_ngram)] + counter[word]
        suf_dict[last_ngram] = suf_dict[last_ngram] + counter[word]
        pref_dict[first_ngram] = pref_dict[first_ngram] + counter[word]

    most_common_pref = sorted(pref_dict.items(), key=lambda pair: pair[1], reverse=True)[0]
    most_common_suf = sorted(suf_dict.items(), key=lambda pair: pair[1], reverse=True)[0]
    most_common_wrap = sorted(suf_pref_dict.items(), key=lambda pair: pair[1], reverse=True)[0]

    pref_merges[most_common_pref[0]] = most_common_pref[1]
    if len(most_common_pref[0]) > 1:
        pref_merges[most_common_pref[0][1:]] = pref_merges[most_common_pref[0][1:]] - most_common_pref[1]

    suf_merges[most_common_suf[0]] = most_common_suf[1]
    if len(most_common_suf[0]) > 1:
        suf_merges[most_common_suf[0][:-1]] = suf_merges[most_common_suf[0][:-1]] - most_common_suf[1]

    wrap_merges[most_common_wrap[0]] = most_common_wrap[1]
    if len(most_common_wrap[0][0]) > 1:
        wrap_merges[(most_common_wrap[0][0][:-1], most_common_wrap[0][1])] = \
            wrap_merges(most_common_wrap[0][0][:-1], most_common_wrap[0][1]) - \
            most_common_wrap[1]
    if len(most_common_wrap[0][1]) > 1:
        wrap_merges[(most_common_wrap[0][0], most_common_wrap[0][1][1:])] = \
            wrap_merges(most_common_wrap[0][0], most_common_wrap[0][1][1:]) - \
            most_common_wrap[1]

    newcounter = {
        (word if len(word) == 1 else
         word if word[0] != most_common_pref[0] and word[-1] != most_common_suf[0] else
         NotImplemented
         ): value for word, value in counter.items()}
    return newcounter, merges


def fuck(counter: Dict[str, int], n: int):
    counter = {tuple(k): v for k, v in counter.items()}
    begin = dict()
    for i in range(n):
        counter, begin = pile_of_shit(counter, begin)
    return counter, begin



def make_idx_map(counter: Dict[str, int], defaults: Dict[str, int]) -> Dict[str, int]:
    keys = sorted(((k, v) for k, v in counter.items()), key=lambda x: x[1], reverse=True)
    keys = map(lambda x: x[0], keys)
    keys = {**defaults, **{k: i + len(defaults) for i, k in enumerate(keys)}}
    return defaultdict(lambda: keys[UNK], keys)


def normalize_corpus(files: strs):

    partial = [], []
    samples = []

    for file in files:
        with open(file, 'r') as i_buffer:
            i_wrapper = i_buffer.__iter__()
            full, partial = get_partial_samples(i_wrapper, partial)
        samples.extend(full)
    return samples


def get_partial_samples(i_wrapper: Iterator[str], current: Tuple[strs, strs]) \
        -> Tuple[Sentences,
                 Tuple[strs, strs]]:

    def is_eos(line_: str) -> bool:
        return line_.startswith('</sentence>')

    words, types = current
    samples = []

    for line in i_wrapper:
        if is_word(line):
            words.append(extract_word(line))
        elif is_type(line):
            types.append(extract_type(line))
        elif is_eos(line):
            pairs = list(zip(words, types))
            samples.append(sentence_preprocess(pairs))
            words, types = [], []
    return samples, (words, types)



