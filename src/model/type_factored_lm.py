from src.utils.imports import *


class TypeFactoredLM(Module):
    def __init__(self, masked_encoder: Module, type_classifier: Module, word_classifier: Module) -> None:
        super(TypeFactoredLM, self).__init__()
        self.MaskedEncoder = masked_encoder
        self.TypeClassifier = type_classifier
        self.WordClassifier = word_classifier

    def forward(self, word_embeddings: FloatTensor, masks: LongTensor) -> Tuple[FloatTensor, FloatTensor]:
        encoded = self.MaskedEncoder(word_embeddings, masks)
        word_predictions = self.WordClassifier(encoded)
        type_predictions = self.TypeClassifier(encoded)
        return word_predictions, type_predictions