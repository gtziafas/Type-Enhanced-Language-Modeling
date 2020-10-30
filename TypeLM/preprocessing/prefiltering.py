from typing import List, Tuple, Any
from nlp_nl.nl_eval.datasets import create_ner, create_ud_lassy_small, create_sonar_pos, create_sonar_ner
from string import punctuation
from re import sub

trans_table = str.maketrans('', '', punctuation)


def remove_punct(x: str) -> str:
    return sub(' +', ' ', x.translate(trans_table).strip())


tasks = [create_ner('./nlp_nl/NER'), create_sonar_pos('./nlp_nl/sonar_tasks', False),
         create_sonar_ner('./nlp_nl/sonar_tasks'), create_ud_lassy_small('./nlp_nl/UD_LASSY_SMALL')]


def get_sent(xs: List[Tuple[str, Any]]) -> str:
    return remove_punct(' '.join([x[0] for x in xs]))


def get_sents(xs: List[List[Tuple[str, Any]]]) -> List[str]:
    return [get_sent(x) for x in xs]


all_sents = set(sum([sum([get_sents(task.train_data), get_sents(task.dev_data), get_sents(task.test_data)], [])
                     for task in tasks], []))
