from src.utils.imports import *
from src.utils.utils import positional_encoding
from src.model.masked_encoder import EncoderLayer, WeightedLayerEncoder


class TypeFactoredLM(Module):
    def __init__(self, masked_encoder: Module, type_classifier: Module, word_classifier: Module) -> None:
        super(TypeFactoredLM, self).__init__()
        self.masked_encoder = masked_encoder
        self.type_classifier = type_classifier
        self.word_classifier = word_classifier

    def forward(self, word_embeds: FloatTensor, msk: LongTensor) -> Tuple[FloatTensor, FloatTensor]:
        batch_size, num_words, d_model = word_embeds.shape[0:3]

        positional_encodings = positional_encoding(b=batch_size, n=num_words, d_model=d_model,
                                                   device=word_embeds.device.__str__())
        word_embeds = word_embeds + positional_encodings
        weighted, final = self.masked_encoder(word_embeds, msk)
        word_preds = self.word_classifier(final)
        type_preds = self.type_classifier(weighted)

        return word_preds, type_preds


class EndToEnd(Module):
    def __init__(self, type_factored_lm: TypeFactoredLM, word_embedder: Module):
        super(EndToEnd, self).__init__()
        self.word_embedder = word_embedder
        self.type_factored_lm = type_factored_lm

    def forward(self, word_ids: LongTensor, mask: LongTensor) -> Tuple[FloatTensor, FloatTensor]:
        word_embeds = self.word_embedder(word_ids)
        return self.type_factored_lm(word_embeds, mask)


def test():
    d_model = 512 
    d_ff = 2048
    d_k, d_v = d_model, d_model
    type_vocab_size, word_vocab_size = 1000, 1000
    num_layers = 6

    encoder = WeightedLayerEncoder(module_maker=EncoderLayer, num_layers=6, num_heads=8,
                                   d_model=d_model, d_k=d_k, d_v=d_v, activation_fn=F.relu)
    type_preds = Linear(d_model, type_vocab_size)
    word_preds = Linear(d_model, word_vocab_size)
    lm = TypeFactoredLM(masked_encoder=encoder, type_classifier=type_preds, word_classifier=word_preds).to('cuda')

    msk = torch.ones(2, 3, 3).to('cuda')
    x = torch.rand((2, 3, d_model)).to('cuda')

    out = lm(x, msk)
    print(out[0].shape, out[1].shape)
