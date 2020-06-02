from typing import Sequence, Tuple, Optional, Callable
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
    def file_to_trees(file: str) -> Sequence[Tuple[str, Tree.ElementTree]]:
        blocks = split_xml(unzip(file))
        blocks, names = list(zip(*blocks))
        names = list(map(lambda name: file+name, names))
        return list(zip(names, list(map(lambda block: Tree.ElementTree(Tree.fromstring(block)), blocks))))

    @staticmethod
    def file_to_projections(file: str, projector: Callable[[Tree.ElementTree], Projections]) -> Projections:
        ret = list(chain.from_iterable([projector(tree) for _, tree in DatasetMaker.file_to_trees(file)]))
        return [r for r in ret if r is not None]

    def iterate_data(self, projector: Callable[[Tree.ElementTree], Projections], save_to: str = './TypeLM/data/dump',
                     start_from: int = 0) -> None:
        for i, file in tqdm(enumerate(self.filelist)):
            if i <= start_from:
                continue
            projections = self.file_to_projections(file, projector)
            strs = '\n'.join(list(map(lambda p: self.project_to_str(*p), projections)))
            with open(f'{save_to}/{i}.dmp', 'w') as dump:
                dump.write(strs)
            with open('./TypeLM/data/last_file.txt', 'w') as g:
                g.write(str(i))

    @staticmethod
    def project_to_str(words: Sequence[str], types: Sequence[str]) -> str:
        word_str = '\t'.join(words)
        type_str = '\t'.join(types)
        return f'{word_str}\n{type_str}\n'


def compose(tree: Tree.ElementTree) -> Projections:
    try:
        dags = transformer(tree)
        if dags is None:
            return []
        dags = [extractor(dag) for dag in dags]
        tuples = [(get_words(dag),
                   [t.polish() for t in get_types(dag)]) for dag in dags if dag is not None]
        return tuples
    except KeyError:
        return []


with open('./TypeLM/data/filelist.txt', 'r') as f:
    filelist = f.read().split('\n')
dsmk = DatasetMaker(None, filelist)
dsmk.iterate_data(compose)
