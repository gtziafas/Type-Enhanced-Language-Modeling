from typing import Iterator, Callable, Any, Tuple, Sequence
from random import sample
from abc import abstractmethod, ABC


Sample = Tuple[Sequence[int], Sequence[int]]
Samples = Sequence[Sample]


def parse(line: str) -> Sample:
    words, types = line.split('\t')
    words = list(map(int, words.split()))
    types = list(map(int, types.split()))
    return words, types


def shuffle_chunk(chunk: Samples) -> Samples:
    return sample(chunk, len(chunk))


class DataLoader(ABC):
    @abstractmethod
    def __next__(self) -> Sample:
        pass

    @abstractmethod
    def get_batch(self) -> Samples:
        pass

    @abstractmethod
    def get_processed_batch(self) -> Any:
        pass


class LazyLoader(DataLoader):
    def __init__(self, filepath: str, chunk_size: int, batch_size: int, post_proc: Callable[[Samples], Any]):
        self.filepath = filepath
        self.line_iterator = open(self.filepath, 'r')
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.chunk = self.get_contiguous_chunk()
        self.post_proc = post_proc

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

    def get_processed_batch(self) -> Any:
        return self.post_proc(self.get_batch())


class EagerLoader(DataLoader):
    def __init__(self, filepath: str, batch_size: int, post_proc: Callable[[Samples], Any]):
        with open(filepath, 'r') as f:
            self.lines = list(map(parse, f.readlines()))
        self.line_iterator = self.make_line_iterator()
        self.batch_size = batch_size
        self.post_proc = post_proc

    def make_line_iterator(self):
        return iter(self.lines)

    def __next__(self) -> Sample:
        return self.line_iterator.__next__()

    def get_batch(self) -> Samples:
        batch = []
        for i in range(self.batch_size):
            try:
                batch.append(self.__next__())
            except StopIteration:
                break
        if i == 0:
            raise StopIteration
        return batch

    def get_processed_batch(self) -> Any:
        return self.post_proc(self.get_batch())



