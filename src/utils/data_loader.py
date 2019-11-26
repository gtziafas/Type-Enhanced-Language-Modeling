from src.utils.imports import *
from src.utils.utils import mask_indices, mask_sampling
import xml.etree.cElementTree as et

from glob import glob
import os
from functools import reduce
import subprocess


Sample = Sequence[Tuple[str, str]]
Samples = Sequence[Sample]


directory = '/run/user/1000/gvfs/smb-share:server=solis-storage01,share=uil-ots$/LASSY 4.0/LassyLarge'


def mask_sample(sample: Sample, random_chance: float = 0.15, mask_token: str = '[MASK]') -> Tuple[Sample, ints]:
    words, types = list(zip(*sample))
    masked_indices = mask_indices(len(words), random_chance)
    # todo. actual sampling
    words = mask_sampling(words, masked_indices, lambda: mask_token)
    types = mask_sampling(types, masked_indices, lambda: mask_token)
    return list(zip(words, types)), masked_indices


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
    def __init__(self, outer_dir: str, extract_fn: Callable[[str], Optional[Sample]]):
        # /run/user/1000/gvfs/smb-share:server=solis-storage01,share=uil-ots$/LASSY 4.0/LassyLarge'
        self.extractor = extract_fn
        inner_dirs = list(map(lambda x: outer_dir + '/' + x, os.listdir(outer_dir)))
        inner_dirs = list(filter(os.path.isdir, inner_dirs))
        self.inner_dirs = list(map(lambda x: x + '/COMPACT/', inner_dirs))
        self.filelist = reduce(lambda x, y: x+y, map(get_files, self.inner_dirs))
        print('Added a total of {} compressed files.'.format(len(self.filelist)))

    def file_to_trees(self, file: str) -> Sequence[et.ElementTree]:
        blocks = split_xml(unzip(file))
        blocks = list(map(lambda block: block[0], blocks))
        return list(map(lambda block: et.ElementTree(et.fromstring(block)), blocks))

    def file_to_projections(self, file: str, projector) \
            -> Sequence[Tuple[Sequence[str], Sequence[str]]]:
        return list(filter(lambda x: x is not None,
                           map(lambda tree: tuple(projector(tree)[0:2]), self.file_to_trees(file))))

dsmk = DatasetMaker(directory, lambda: None)
