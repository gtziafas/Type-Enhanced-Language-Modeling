from nlp_nl.nl_eval.task import TokenTask
from TypeLM.preprocessing.tokenizer import Tokenizer, WordTokenizer
from typing import List, Tuple


def tokenize_data(tokenizer: Tokenizer, data: List[List[Tuple[str, int]]], pad: int) \
        -> List[Tuple[List[int], List[int]]]:
    unzipped = [list(zip(*datum)) for datum in data]
    wss, tss = list(zip(*unzipped))
    return [tokenize_token_pairs(tokenizer.word_tokenizer, ws, ts, pad) for ws, ts in zip(wss, tss)]


def tokenize_token_pairs(wtokenizer: WordTokenizer, words: List[str], tokens: List[int], pad: int) -> \
        Tuple[List[int], List[int]]:
    assert len(words) == len(tokens)
    words = [wtokenizer.core.tokenize(w) for w in words]
    tokens = [[t] + [pad] * (len(w) - 1) for w, t in zip(words, tokens)]
    words = sum(words, [])
    tokens = sum(tokens, [])
    assert len(words) == len(tokens)
    word_ids = wtokenizer.core.convert_tokens_to_ids(words)
    return [wtokenizer.core.cls_token_id] + word_ids + [wtokenizer.core.sep_token_id], [pad] + tokens + [pad]
