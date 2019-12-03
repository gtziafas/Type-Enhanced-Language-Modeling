from collections import Counter
from typing import Sequence, Iterable, Dict, Tuple

from functools import reduce 
from operator import add  

from threading import Thread


def merge_dicts(dicts: Iterable[Dict[str, int]]) -> Dict[str, int]:
	return reduce(add, dicts, dict())


def sanity(word: str) -> str:
	return word


def get_vocabs_one_file(file: str) -> Tuple[Dict[str, int], Dict[str, int]]:
	with open(file, 'r') as fp:
		lines = fp.readlines()

	words = filter(lambda line: line.startswith('\t\t<word>"'), lines)
	words = map(lambda line: (line.split('<word>"')[1]).split('"</word>')[0], words)
	words = map(lambda word: sanity(word), words)

	types = filter(lambda line: line.startswith('\t\t<type>"') , lines)
	types = map(lambda line: (line.split('<type>"')[1]).split('"</type>')[0], types)

	words = Counter(words)
	types = Counter(types)

	return words, types


def get_vocabs_one_thread(files: Sequence[str], start: int, end: int) -> Tuple[Dict[str, int], Dict[str, int]]:
	dicts = map(get_vocabs_one_file, files[start:end])
	words, types = map(merge_dicts, zip(*dicts))
	return words, types


def get_vocabs_multi_thread(files: Sequence[str], nthr: int = 2) -> Tuple[Dict[str, int], Dict[str, int]]:
	threads = []
	step = len(files) // ntrhr
	chunks = [files[x*step:(x+1)*step] for x in range(0, nthr)]
	for idx in range(nthr):
		t = Thread(target=get_vocabs_one_thread, args=(chunks[idx], idx, ))
		threads.append(t)
		t.start()

	for idx in range(nthr):
		t.join()


def go():
	fs = list(range(100))
	fs = ['x0'+str(idx) if idx<10 else 'x' + str(idx) for idx in fs]
	words, types = get_vocabs_one_thread(fs, 0, len(fs))

	print(len(words))
	print(len(types))

	return words, types


if __name__ == "__main__":
	go()
