from src.utils.imports import *
from src.utils.utils import positional_encoding
from src.model.masked_encoder import make_encoder, EncoderLayer, DoubleOutputEncoder


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
        intermediate, final = self.masked_encoder((word_embeds, msk))
        word_preds = self.word_classifier(final[0])
        type_preds = self.type_classifier(intermediate[0])

        return word_preds, type_preds


def test():
    d_model = 512 
    d_ff = 2048
    d_k, d_v = d_model, d_model
    type_vocab_size, word_vocab_size = 1000, 1000
    num_layers = 6

    encoder = DoubleOutputEncoder(module_maker=EncoderLayer, bottom_depth=3, top_depth=3, num_heads=8,
                                  d_model=d_model, d_k=d_k, d_v=d_v, activation_fn=F.relu)
    type_preds = Linear(d_model, type_vocab_size)
    word_preds = Linear(d_model, word_vocab_size)
    lm = TypeFactoredLM(masked_encoder=encoder, type_classifier=type_preds, word_classifier=word_preds).to('cuda')

    msk = torch.ones(2, 3, 3).to('cuda')
    x = torch.rand((2, 3, d_model)).to('cuda')

    out = lm(x, msk)
    print(out[0].shape, out[1].shape)
