from src.utils.imports import *
from src.utils.utils import PositionalEncoding
from src.model.masked_encoder import MaskedEncoder, MaskedEncoderLayer

class TypeFactoredLM(Module):
    def __init__(self, masked_encoder: Module, type_classifier: Module, word_classifier: Module) -> None:
        super(TypeFactoredLM, self).__init__()
        self.masked_encoder = masked_encoder
        self.type_classifier = type_classifier
        self.word_classifier = word_classifier

    def forward(self, word_embeds: FloatTensor, msk: LongTensor) -> Tuple[FloatTensor, FloatTensor]:
        batch_size, num_words, d_model = word_embeds.shape[0:3]

        word_embeds = PositionalEncoding(b=batch_size, n=num_words, d_model=d_model, device=word_embeds.device) + word_embeds 
        encoded = self.masked_encoder((word_embeds, msk))[0]
        word_preds = self.word_classifier(encoded)
        type_preds = self.type_classifier(encoded)
        
        return word_preds, type_preds

def test():
    d_model = 512 
    d_ff = 2048
    d_k, d_v = d_model, d_model
    type_vocab_size, word_vocab_size = 10, 10
    num_layers = 6
    encoder = MaskedEncoder( MaskedEncoderLayer, num_layers=num_layers, num_heads=8, d_model=d_model, d_k=d_k, d_v=d_v)
    type_preds = Linear(d_model, type_vocab_size)
    word_preds = Linear(d_model, word_vocab_size)
    lm = TypeFactoredLM(masked_encoder=encoder, type_classifier=type_preds, word_classifier=word_preds).to('cuda')
    msk = torch.ones(2,3,3).to('cuda')
    x = torch.rand( (2,3,512)).to('cuda')

    out = lm(x,msk)
    print(out[0].shape, out[1].shape)
