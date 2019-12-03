import random
import string 
import xml.etree.cElementTree as et

from typing import Tuple, Sequence, Optional, List

filename = 'temp.xml'

def random_string(max_len: int=15) -> str:
	letters = string.ascii_lowercase
	length = random.randint(1, max_len)
	return ''.join(random.choice(letters) for _ in range(length))

def random_sequence(max_seq_len: int=15, fixed: Optional[int]=None) -> Sequence[str]:
	length = random.randint(1, max_seq_len) if fixed is None else fixed 
	return [random_string() for _ in range(length)]

def random_sample(max_len: int=15) -> Tuple[str, Sequence[str], Sequence[str]]:
	name = random_string()
	rnd = random.randint(1, max_len)
	words = random_sequence(fixed=rnd)
	types = random_sequence(fixed=rnd)
	
	return (name, words, types)
	
def random_samples(num_samples: int) -> List[Tuple[str, Sequence[str], Sequence[str]]]:
	return [random_sample() for _ in range(num_samples)]

dataset = random_samples(100)
dataset[0][1][0] += '"'

def wrap_to_xml(src: str) -> str:
	start = '<?xml version="1.0"?>\n' + '<sentences>\n'
	end = '\n</sentences>'
	return start + src + end 

def projection_to_xml(name: str, words: Sequence[str], types: Sequence[str]) -> str:
	word_str = '\t<words>\n\t\t' + \
	           '\n\t\t'.join(list(map(lambda word: '<word>"' + word + '"</word>', words))) + \
	           '\n\t</words>'
	type_str = '\t<types>\n\t\t' \
	           + '\n\t\t'.join(list(map(lambda type: '<type>"' + type + '"</type>', types))) + \
	           '\n\t</types>'

	return '<sentence id="' + name + '">' + '\n' + word_str + '\n' + type_str + '\n</sentence>'


def merge_xmls(src: str, news: List[Tuple[str, Sequence[str], Sequence[str]]]) -> str:
	return src + '\n' + '\n'.join([projection_to_xml(*news[i]) for i in range(len(news))])  


entire = wrap_to_xml( merge_xmls('', dataset))
print(entire)

('/run/user/1000/gvfs/smb-share:server=solis-storage01,share=uil-ots$/LASSY 4.0/LassyLarge/EINDHOVEN/COMPACT/cdb.data.dz1_1', [('in verband met', 'de', 'gemiddeld', 'langere', 'levensduur', 'van', 'de', 'vrouw'), (<NP> obj1 → PP, <N> invdet → NP, <<NP> mod → NP> mod → <NP> mod → NP, <NP> mod → NP, N, <NP> obj1 → <NP> mod → NP, <N> invdet → NP, N)])