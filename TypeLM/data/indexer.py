from TypeLM.data.tokenizer import Tokenizer
from itertools import product
import pickle


class Indexer(object):
    """A class that indexes the training data using the tokeniser."""
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.word2index = create_word_indices()

    def create_word_indices(self):
        all_input_tokens_word = list(self.tokenizer.vocab) + \
                                self.tokenizer.wraps + \
                                self.tokenizer.prefixes + \
                                self.tokenizer.suffixes + \
                                self.tokenizer.tokens
        word2index = {w: i for i, w in enumerate(self.tokenizer)}
        with open('./TypeLM/data/indexer_data.p', 'wb') as f:
            pickle.dump(word2index, f)

    def create_type_indices(self):
        pass

    def index_word(self, word: str):



    def index_type(self, type: str):
