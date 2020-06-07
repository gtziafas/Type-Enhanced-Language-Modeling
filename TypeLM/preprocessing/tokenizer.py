from transformers import BertTokenizer
from typing import List, Set, Tuple, Optional
from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import chain

ints = List[int]
strs = List[str]


class WordTokenizer(object):
    def __init__(self):
        self.core = BertTokenizer.from_pretrained('bert-base-dutch-cased')

    def __len__(self):
        return self.core.vocab_size

    def convert_sent_to_ids(self, sent: str, **kwargs) -> ints:
        return self.core.encode(sent, **kwargs)

    def convert_sent_to_tokens(self, sent: str, **kwargs) -> strs:
        return self.core.tokenize(sent, **kwargs)

    def convert_ids_to_tokens(self, ids: ints) -> strs:
        return self.core.convert_ids_to_tokens(ids)

    def convert_sent_to_ids_and_wordstarts(self, sent: str) -> Tuple[ints, ints]:
        tokens: strs = self.core.tokenize(sent)
        word_starts: ints = [1 if not t.startswith('##') else 0 for i, t in enumerate(tokens)]
        tokens = [self.core.cls_token] + tokens + [self.core.sep_token]
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
        self.special_tokens = ['[PAD]', '[UNK]']
        vocabulary = {k: i for i, k in enumerate(self.special_tokens + sorted(vocabulary))}
        self.vocabulary = defaultdict(lambda: vocabulary['[UNK]'], vocabulary)
        self.inverse_vocabulary = {v: k for k, v in self.vocabulary.items()}
        self.PAD_TOKEN = '[PAD]'
        self.UNK_TOKEN = '[UNK]'
        self.PAD_TOKEN_ID = self.vocabulary['[PAD]']
        self.UNK_TOKEN_ID = self.vocabulary['[UNK]']

    def __len__(self) -> int:
        return len(self.vocabulary)

    def convert_types_to_ids(self, types: strs) -> ints:
        return [self.vocabulary[atom] for atom in types]

    def convert_ids_to_types(self, ids: ints) -> strs:
        return [self.inverse_vocabulary[_id] for _id in ids]


class Tokenizer(object):
    def __init__(self, type_vocabulary: Set[str], atomic: bool):
        self.word_tokenizer = WordTokenizer()
        self.type_tokenizer = AtomTokenizer(type_vocabulary) if atomic else FullTokenizer(type_vocabulary)

    def convert_sent_to_ids(self, sent: str, **kwargs) -> ints:
        return self.word_tokenizer.convert_sent_to_ids(sent, **kwargs)

    def convert_types_to_ids(self, types: strs) -> ints:
        return self.type_tokenizer.convert_types_to_ids(types)

    def convert_pair_to_ids(self, sent: str, types: strs, max_wlen: int, max_tlen: int, min_wlen: int, min_tlen: int,
                            ) -> Optional[Tuple[ints, ints]]:
        if isinstance(self.type_tokenizer, FullTokenizer):
            s_ids, word_starts = self.word_tokenizer.convert_sent_to_ids_and_wordstarts(sent)
            type_iter = iter(self.convert_types_to_ids(types))
            t_ids = [self.type_tokenizer.PAD_TOKEN_ID if word_starts[i] == 0 else type_iter.__next__()
                     for i in range(len(s_ids))]
        elif isinstance(self.type_tokenizer, AtomTokenizer):
            s_ids = self.convert_sent_to_ids(sent)
            t_ids = self.convert_types_to_ids(types)
        else:
            raise NotImplementedError
        return (s_ids, t_ids) if min_wlen < len(s_ids) < max_wlen and min_tlen < len(t_ids) < max_tlen else None


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


def _parse_dump_atomic(dump: str = './TypeLM/data/extraction/dump',):
    with open('./TypeLM/data/indexing/atomset.txt', 'r') as f:
        atomset = set(f.read().split('\n'))
    tokenizer = Tokenizer(type_vocabulary=atomset, atomic=True)
    with open(dump, 'r') as f:
        with open('./TypeLM/data/indexing/atomic_dump', 'a') as g:
            while True:
                next_line = f.__next__().strip('\n')
                if next_line == '':
                    sent = f.__next__().strip('\n')
                else:
                    sent = next_line
                types = _parse_line(f.__next__())
                atoms = list(chain.from_iterable([t.split() + [tokenizer.type_tokenizer.SEP_TOKEN] for t in types]))
                tmp = tokenizer.convert_pair_to_ids(sent, atoms, max_wlen=50, max_tlen=200, min_wlen=1, min_tlen=-1)
                if tmp is None:
                    continue
                s_ids = ' '.join(map(str, tmp[0]))
                t_ids = ' '.join(map(str, tmp[1]))
                line = f'{s_ids}\t{t_ids}\n'
                g.write(line)
