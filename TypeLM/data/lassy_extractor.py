from typing import Sequence, Tuple, Optional, Callable, Iterator
import xml.etree.cElementTree as Tree

from glob import glob
import os
from functools import reduce
import subprocess

from itertools import chain
from tqdm import tqdm

from LassyExtraction.utils.tools import get_types, get_words
from LassyExtraction.extraction import extractor
from LassyExtraction.transformations import transformer

import signal

_directory = '/run/user/1000/gvfs/smb-share:server=solis-storage01,share=uil-ots$/LASSY 4.0/LassyLarge'


Projection = Tuple[Sequence[str], Sequence[str]]
Projections = Sequence[Projection]


def unzip(file: str) -> str:
    bytes_ = subprocess.Popen('dictunzip -c ' + '"' + file + '"', shell=True, stdout=subprocess.PIPE).stdout.read()
    return bytes_.decode('utf-8')


def split_xml(uncompressed: str) -> Sequence[Tuple[str, str]]:
    def getname(xml_block: str) -> str:
        """
            Finds the sentence name attribute of an xml part.
        :param xml_block:
        :type xml_block:
        :return:
        :rtype:
        """
        return xml_block.split('sentid="')[1].split('"')[0]

    xmls = uncompressed.split('</alpino_ds>\n')[:-1]
    xmls = list(map(lambda x: x + '</alpino_ds>', xmls))
    names = list(map(getname, xmls))
    return list(zip(xmls, names))


def get_files(inner_dir: str) -> Sequence[str]:
    filelist = [y for x in os.walk(inner_dir) for y in glob(os.path.join(x[0], '*.[dD][zZ]'))]
    print('Added {} files from subdir {}'.format(len(filelist), inner_dir.split('/')[-3]))
    return filelist


class DatasetMaker(object):
    def __init__(self, outer_dir: Optional[str], filelist: Optional[Sequence[str]]):
        if filelist is None and outer_dir is None:
            raise ValueError('Provide either an outer dir or a filelist.')
        elif filelist is None:
            print('Crawling..')
            inner_dirs = list(map(lambda x: outer_dir + '/' + x, os.listdir(outer_dir)))
            inner_dirs = list(filter(os.path.isdir, inner_dirs))
            self.inner_dirs = list(map(lambda x: x + '/COMPACT/', inner_dirs))
            filelist = reduce(lambda x, y: x+y, map(get_files, self.inner_dirs))
            self.filelist = sorted(filelist)
        else:
            self.filelist = filelist
        print('Added a total of {} compressed files.'.format(len(self.filelist)))

    @staticmethod
    def file_to_trees(file: str) -> Iterator[Tree.ElementTree]:
        blocks = split_xml(unzip(file))
        if not blocks:
            raise ConnectionAbortedError('Failed to connect.')
        blocks, _ = list(zip(*blocks))
        return map(lambda block: Tree.ElementTree(Tree.fromstring(block)), blocks)

    @staticmethod
    def file_to_projections(file: str, projector: Callable[[Tree.ElementTree], Projections]) -> Iterator[Projection]:
        return filter(lambda p: p is not None,
                      chain.from_iterable(map(projector,
                                              DatasetMaker.file_to_trees(file))))

    def iterate_data(self, projector: Callable[[Tree.ElementTree], Projections], save_to: str = './TypeLM/data/dump',
                     file_id: int = 0) -> None:
        with open(save_to, 'a') as dump:
            for i, file in tqdm(enumerate(self.filelist)):
                if i < file_id:
                    continue
                print(f'Opening {i}')
                projections = self.file_to_projections(file, projector)
                strs = map(lambda p: self.project_to_str(*p), projections)
                dump.write('\n'.join(strs))
                with open('./TypeLM/data/last_file', 'w') as g:
                    g.write(str(i+1))

    @staticmethod
    def project_to_str(words: Sequence[str], types: Sequence[str]) -> str:
        word_str = '\t'.join(words)
        type_str = '\t'.join(types)
        return f'{word_str}\n{type_str}\n'


def timeout(t: int):
    def handler(signum, frame):
        raise TimeoutError('Took too long.')

    def decorator(fn):
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(t)
            try:
                ret = fn(*args, **kwargs)
                signal.alarm(0)
                return ret
            except TimeoutError:
                return []
            except KeyError:
                return []
            except AttributeError:
                return []
            except IndexError:
                return []
        return wrapper
    return decorator


@timeout(t=5)
def compose(tree: Tree.ElementTree) -> Projections:
    dags = transformer(tree)
    if dags is None:
        return []
    dags = [extractor(dag) for dag in dags]
    tuples = [(get_words(dag),
               [t.polish() for t in get_types(dag)]) for dag in dags if dag is not None]
    return tuples


with open('./TypeLM/data/filelist.txt', 'r') as f:
    filelist = f.read().split('\n')
dsmk = DatasetMaker(None, filelist)
dsmk.iterate_data(compose, file_id=5250)
