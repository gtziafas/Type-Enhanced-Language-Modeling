from typing import Tuple, List, Iterator
from random import sample

Sample = Tuple[List[int], List[int]]
Samples = List[Sample]


def parse(line: str) -> Sample:
    words, types = line.split('\t')
    words = list(map(int, words.split()))
    types = list(map(int, types.split()))
    return words, types


def shuffle_chunk(chunk: Samples) -> Samples:
    return sample(chunk, len(chunk))


class DataLoader(object):
    def __init__(self, filepath: str, chunk_size: int, batch_size: int):
        self.filepath = filepath
        self.line_iterator = open(self.filepath, 'r')
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.chunk = self.get_contiguous_chunk()

    def get_next_line(self) -> str:
        try:
            return self.line_iterator.__next__()
        except StopIteration:
            self.line_iterator = open(self.filepath, 'r')
            return self.line_iterator.__next__()

    def get_contiguous_chunk(self) -> Iterator[Sample]:
        return iter(shuffle_chunk([parse(self.get_next_line()) for _ in range(self.chunk_size)]))

    def __next__(self) -> Sample:
        try:
            return self.chunk.__next__()
        except StopIteration:
            self.chunk = self.get_contiguous_chunk()
            return self.chunk.__next__()

    def get_batch(self) -> Samples:
        return [self.__next__() for _ in range(self.batch_size)]
