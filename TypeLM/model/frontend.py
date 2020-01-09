from TypeLM.utils.imports import *
from TypeLM.model.train import pad_sequence
from TypeLM.model.type_factored_lm import TypeFactoredLM
from TypeLM.data.tokenizer import Tokenizer, Indexer


class Model(object):
    def __init__(self, backend: TypeFactoredLM, tokenizer: Tokenizer, indexer: Indexer, device: str):
        self.backend = backend
        self.tokenizer = tokenizer
        self.indexer = indexer
        self.ints_to_types = {v: k for k, v in self.indexer.type_indices.items()}
        self.device = device
        self.backend.eval()

    def _supertag(self, word_ids: LongTensor, pad_mask: LongTensor) -> List[List[str]]:
        type_probabilities = self.backend.forward(word_ids, pad_mask)[1]
        type_probabilities = torch.argmax(type_probabilities, dim=-1).tolist()
        return list(map(lambda sentence:
                        list(map(lambda idx: self.ints_to_types[idx], sentence)), type_probabilities))

    def supertag(self, sentences: List[str], tokenize: bool = True) -> List[List[Tuple[str, str]]]:
        sentences = list(map(lambda sentence: self.tokenizer.tokenize_sentence(sentence, add_eos=True), sentences))
        indices = list(map(self.indexer.index_sentence, sentences))
        lens = list(map(len, sentences))
        indices = list(map(LongTensor, indices))
        indices = pad_sequence(indices).to(self.device)
        word_pads = torch.ones(indices.shape[0], indices.shape[1], indices.shape[1], device=self.device)
        for i, l in enumerate(lens):
            word_pads[i, :, l::] = 0
        supertags = self._supertag(indices, word_pads)
        st = list(zip(sentences, supertags))
        return [list(zip(*st)) for st in zip(sentences, supertags)]


def test():
    from main import default_model, default_tokenizer, Indexer
    t = default_tokenizer()
    i = Indexer(t)
    return Model(default_model(), t, i, 'cuda')