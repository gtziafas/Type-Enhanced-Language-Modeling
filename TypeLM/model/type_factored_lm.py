from TypeLM.utils.imports import *
from TypeLM.utils.utils import positional_encoding, count_parameters
from TypeLM.model.masked_encoder import EncoderLayer, WeightedLayerEncoder


class TypeFactoredLM(Module):
    def __init__(self, masked_encoder: Module, type_classifier: Module,
                 masked_encoder_kwargs: Dict, num_words: int, type_classifier_kwargs: Dict,
                 padding_idx: int = 0) -> None:
        super(TypeFactoredLM, self).__init__()
        self.masked_encoder = masked_encoder(**masked_encoder_kwargs)
        self.type_classifier = type_classifier(**type_classifier_kwargs)
        self.word_embedder = Embedding(num_embeddings=num_words, embedding_dim=masked_encoder_kwargs['d_model'],
                                       padding_idx=padding_idx)

    def forward(self, word_ids: LongTensor, msk: LongTensor) -> Tuple[Tensor, Tensor]:
        word_embeds = self.word_embedder(word_ids)
        batch_size, num_words, d_model = word_embeds.shape[0:3]
        positional_encodings = positional_encoding(b=batch_size, n=num_words, d_model=d_model,
                                                   device=word_embeds.device.__str__())

        word_embeds = word_embeds + positional_encodings
        weighted, final = self.masked_encoder(word_embeds, msk)

        word_preds = self.word_classifier(final)
        type_preds = self.type_classifier(weighted)

        return word_preds, type_preds

    def word_classifier(self, final: Tensor) -> Tensor:
        return final@self.word_embedder.weight.transpose(1, 0)


def test():
    d_model = 512 
    d_ff = 2048
    d_k, d_v = d_model, d_model
    type_vocab_size, word_vocab_size = 5000, 500000
    num_layers = 6

    encoder_params = {'module_maker': EncoderLayer,
                      'num_layers': num_layers,
                      'num_heads': 8,
                      'd_model': d_model,
                      'd_ff': d_ff,
                      'd_k': d_k,
                      'd_v': d_v,
                      'activation_fn': F.gelu}
    type_pred_params = {'in_features': d_model, 'out_features': type_vocab_size}

    lm = TypeFactoredLM(masked_encoder=WeightedLayerEncoder,
                        type_classifier=Linear,
                        num_words=word_vocab_size,
                        masked_encoder_kwargs=encoder_params,
                        type_classifier_kwargs=type_pred_params,
                        ).to('cuda')

    print(count_parameters(lm))

    msk = torch.ones(2, 3, 3).to('cuda')
    x = torch.randint(size=(2, 3), low=0, high=word_vocab_size).to('cuda')

    out = lm(x, msk)
    print(out[0].shape, out[1].shape)
