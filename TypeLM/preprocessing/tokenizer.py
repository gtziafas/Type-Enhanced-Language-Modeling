from transformers import BertTokenizer
from typing import List, Set, Tuple, Optional
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from itertools import chain
import pickle

ints = List[int]
strs = List[str]


class WordTokenizer:
    def __init__(self):
        self.core = BertTokenizer.from_pretrained('wietsedv/bert-base-dutch-cased')

    def __len__(self):
        return self.core.vocab_size

    def convert_sent_to_ids(self, sent: str, **kwargs) -> ints:
        return self.core.encode(sent, **kwargs)

    def convert_sent_to_tokens(self, sent: str, **kwargs) -> strs:
        return self.core.tokenize(sent, **kwargs)

    def convert_ids_to_tokens(self, ids: ints) -> strs:
        return self.core.convert_ids_to_tokens(ids)

    def convert_sent_to_ids_and_wordstarts(self, sent: str) -> Tuple[ints, ints]:
        words = self.core.prepare_for_tokenization(sent).split()
        tokens = [self.core.tokenize(word) for word in words]
        word_starts = sum([[1] + [0] * (len(t) - 1) for t in tokens], [])
        tokens = [self.core.cls_token] + sum(tokens, []) + [self.core.sep_token]
        return self.core.convert_tokens_to_ids(tokens), [0] + word_starts + [0]


class TypeTokenizer(ABC):
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def convert_types_to_ids(self, types: strs) -> ints:
        pass

    @abstractmethod
    def convert_ids_to_types(self, ids: ints) -> strs:
        pass


class AtomTokenizer(TypeTokenizer):
    def __init__(self, vocabulary: Set[str]):
        self.special_tokens = ['[PAD]', '[SOS]', '[SEP]']
        self.vocabulary = {k: i for i, k in enumerate(self.special_tokens + sorted(vocabulary))}
        self.inverse_vocabulary = {v: k for k, v in self.vocabulary.items()}
        self.PAD_TOKEN = '[PAD]'
        self.SOS_TOKEN = '[SOS]'
        self.SEP_TOKEN = '[SEP]'
        self.PAD_TOKEN_ID = self.vocabulary['[PAD]']
        self.SOS_TOKEN_ID = self.vocabulary['[SOS]']
        self.SEP_TOKEN_ID = self.vocabulary['[SEP]']

    def __len__(self) -> int:
        return len(self.vocabulary)

    def convert_types_to_ids(self, types: strs) -> ints:
        return [self.vocabulary[atom] for atom in types]

    def convert_ids_to_types(self, ids: ints) -> strs:
        return [self.inverse_vocabulary[_id] for _id in ids]


class FullTokenizer(TypeTokenizer):
    def __init__(self, vocabulary: Set[str]):
        self.special_tokens = ['[PAD]', '[UNK]', '[MWU]']
        vocabulary = {k: i for i, k in enumerate(self.special_tokens + sorted(vocabulary))}
        self.vocabulary = defaultdict(lambda: vocabulary['[UNK]'], vocabulary)
        self.inverse_vocabulary = {v: k for k, v in self.vocabulary.items()}
        self.PAD_TOKEN = '[PAD]'
        self.UNK_TOKEN = '[UNK]'
        self.MWU_TOKEN = '[MWU]'
        self.PAD_TOKEN_ID = self.vocabulary['[PAD]']
        self.UNK_TOKEN_ID = self.vocabulary['[UNK]']
        self.MWU_TOKEN_ID = self.vocabulary['[MWU]']

    def __len__(self) -> int:
        return len(self.vocabulary)

    def convert_types_to_ids(self, types: strs) -> ints:
        return [self.vocabulary[atom] for atom in types]

    def convert_ids_to_types(self, ids: ints) -> strs:
        return [self.inverse_vocabulary[_id] for _id in ids]


class Tokenizer:
    def __init__(self, type_vocabulary: Set[str], atomic: bool):
        self.word_tokenizer = WordTokenizer()
        self.type_tokenizer = AtomTokenizer(type_vocabulary) if atomic else FullTokenizer(type_vocabulary)

    def convert_sent_to_ids(self, sent: str, **kwargs) -> ints:
        return self.word_tokenizer.convert_sent_to_ids(sent, **kwargs)

    def convert_types_to_ids(self, types: strs) -> ints:
        return self.type_tokenizer.convert_types_to_ids(types)

    def convert_pair_to_ids(self, sent: str, types: strs) -> Optional[Tuple[ints, ints]]:
        if isinstance(self.type_tokenizer, FullTokenizer):
            words = sent.split('\t')
            words = [w.split() for w in words]
            types = [[t] + [self.type_tokenizer.MWU_TOKEN] * (len(w)-1) for w, t in zip(words, types)]
            types = sum(types, [])
            s_ids, word_starts = self.word_tokenizer.convert_sent_to_ids_and_wordstarts(sent)
            t_ids = self.convert_types_to_ids(types)
            type_iter = iter(t_ids)
            t_ids = [self.type_tokenizer.PAD_TOKEN_ID if ws == 0 else type_iter.__next__()
                     for ws in word_starts]
            return s_ids, t_ids
        else:
            raise NotImplementedError


def _parse_line(line_: str) -> strs:
    return line_.strip('\n').split('\t')


def _make_atom_set(dump: str = './TypeLM/data/dump') -> Set[str]:
    atoms = set()
    with open(dump, 'r') as f:
        for i in range(20000):
            _ = f.__next__()
            types = _parse_line(f.__next__())
            _ = f.__next__()
            atoms = atoms.union(chain.from_iterable([t.split() for t in types]))
    return atoms


def _make_type_set(dump: str = './TypeLM/data/extraction/dump') -> None:
    with open(dump, 'r') as f:
        counter = defaultdict(lambda: 0)
        while True:
            try:
                next_line = f.__next__().strip('\n')
                if next_line == '':
                    _ = f.__next__().strip('\n')
                else:
                    _ = next_line
                types = _parse_line(f.__next__())
                for t in types:
                    counter[t] += 1
            except StopIteration:
                break
        with open('./TypeLM/data/indexing/typeset.p', 'wb') as g:
            pickle.dump(dict(counter), g)
        typeset = sorted(counter.items(), key=lambda pair: pair[1], reverse=True)
        with open('./TypeLM/data/indexing/typeset.txt', 'w') as g:
            g.write('\n'.join([f'{str(fst)}\t{str(snd)}' for fst, snd in typeset]))


def _make_small_type_set(type_dump: str = './TypeLM/data/indexing/typeset.txt', inclusion: float = 0.95) -> Set[str]:
    with open(type_dump, 'r') as f:
        types = list(map(lambda x: x.split('\t'), f.read().split('\n')))

    total = sum(map(lambda x: int(x[1]), types))
    curr = 0
    kept = set()

    for item, count in types:
        if curr / total >= inclusion:
            break
        curr += int(count)
        kept.add(item)
    print(f'Last item has {count} occurrences.')
    with open('./TypeLM/data/indexing/small_typeset.txt', 'w') as f:
        f.write('\n'.join(sorted(kept)))
    return kept


def _parse_dump_atomic(dump: str = './TypeLM/data/extraction/dump', start_from: int = 0):
    with open('./TypeLM/data/indexing/atomset.txt', 'r') as f:
        atomset = set(f.read().split('\n'))
    tokenizer = Tokenizer(type_vocabulary=atomset, atomic=True)
    with open(dump, 'r') as f:
        read = 0
        with open('./TypeLM/data/indexing/atomic_dump', 'a') as g:
            while True:
                next_line = f.__next__().strip('\n')
                if next_line == '':
                    sent = f.__next__().strip('\n')
                else:
                    sent = next_line
                types = _parse_line(f.__next__())

                read += 1
                if read <= start_from:
                    continue

                atoms = list(chain.from_iterable([t.split() + [tokenizer.type_tokenizer.SEP_TOKEN] for t in types]))
                s_ids_int, t_ids_int = tokenizer.convert_pair_to_ids(sent, atoms)
                s_ids = list(map(str, s_ids_int))
                t_ids = list(map(str, t_ids_int))
                line = f'{s_ids}\t{t_ids}\n'
                g.write(line)

                with open('./TypeLM/data/indexing/last_file', 'w') as h:
                    h.write(str(read + 1))


def _parse_dump_full(dump: str = './TypeLM/data/extraction/dump', start_from: int = 3774747) -> None:
    with open('./TypeLM/data/indexing/small_typeset.txt', 'r') as f:
        typeset = set(f.read().split('\n'))
    tokenizer = Tokenizer(type_vocabulary=typeset, atomic=False)
    with open(dump, 'r') as f:
        read = 0
        with open('./TypeLM/data/indexing/full_dump', 'a') as g:
            while True:
                next_line = f.__next__().strip('\n')
                if next_line == '':
                    sent = f.__next__().strip('\n')
                else:
                    sent = next_line
                types = _parse_line(f.__next__())

                read += 1
                if read <= start_from:
                    continue
                with open('./TypeLM/data/indexing/last_file', 'w') as h:
                    h.write(str(read + 1))
                s_ids_int, t_ids_int = tokenizer.convert_pair_to_ids(sent, types)
                if len(s_ids_int) != len(t_ids_int):
                    print(f'failed at {read}')
                    print(sent)
                    print(types)
                    continue
                s_ids = list(map(str, s_ids_int))
                t_ids = list(map(str, t_ids_int))
                line = f'{" ".join(s_ids)}\t{" ".join(t_ids)}\n'
                g.write(line)


