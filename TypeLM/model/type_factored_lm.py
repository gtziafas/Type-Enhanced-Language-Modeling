from TypeLM.utils.imports import *
from TypeLM.utils.utils import PositionalEncoder, count_parameters
from TypeLM.model.masked_encoder import EncoderLayer, LayerWeighter, Encoder


class TypeFactoredLM(Module):
    def __init__(self, masked_encoder: Module, type_classifier: Module, type_embedder: Module,
                 fusion: Module, masked_encoder_kwargs: Dict, num_words: int, type_classifier_kwargs: Dict,
                 type_embedder_kwargs: Dict, fusion_kwargs: Dict, padding_idx: int = 0,
                 dropout_rate: float = 0.1) -> None:
        super(TypeFactoredLM, self).__init__()
        self.masked_encoder = masked_encoder(**masked_encoder_kwargs)
        self.layer_weighter = LayerWeighter(masked_encoder_kwargs['num_layers']-2)
        self.type_classifier = type_classifier(**type_classifier_kwargs)
        self.type_embedder = type_embedder(**type_embedder_kwargs)
        self.fusion = fusion(**fusion_kwargs)
        self.word_embedder = Embedding(num_embeddings=num_words, embedding_dim=masked_encoder_kwargs['d_model'],
                                       padding_idx=padding_idx)
        self.positional_encoder = PositionalEncoder(dropout_rate)
        self.dropout = Dropout(dropout_rate)

    def forward(self, word_ids: LongTensor, pad_mask: LongTensor) -> Tuple[Tensor, Tensor]:
        layer_outputs = self.get_all_vectors(word_ids, pad_mask)
        weighted = self.layer_weighter(layer_outputs[1:-2])
        type_preds = self.type_classifier(self.dropout(weighted))
        type_preds_activated = type_preds.softmax(dim=-1)
        type_embeddings = self.dropout(self.type_embedder(type_preds_activated))
        word_preds = self.word_classifier(self.fusion(type_embeddings, layer_outputs[-1]))
        return word_preds, type_preds

    def forward_st(self, word_ids: LongTensor, pad_mask: LongTensor) -> Tensor:
        layer_outputs = self.get_all_vectors(word_ids, pad_mask)
        weighted = self.layer_weighter(layer_outputs[1:-2])
        type_preds = self.type_classifier(weighted)
        return type_preds

    def forward_lm(self, word_ids: LongTensor, pad_mask: LongTensor) -> Tensor:
        return self.forward(word_ids, pad_mask)[0]

    def get_all_vectors(self, word_ids: LongTensor, pad_mask: LongTensor) -> Tensor:
        word_embeds = self.word_embedder(word_ids)
        batch_size, num_words, d_model = word_embeds.shape[0:3]
        positional_encodings = self.positional_encoder(b=batch_size, n=num_words, d_model=d_model,
                                                       device=word_embeds.device.__str__())

        word_embeds = word_embeds + positional_encodings
        return self.masked_encoder.forward_all(word_embeds, pad_mask)

    def get_last_vectors(self, word_ids: LongTensor, pad_mask: LongTensor) -> Tensor:
        layer_outputs = self.get_all_vectors(word_ids, pad_mask)
        return layer_outputs[-1]

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
                      'activation_fn': F.relu}
    type_pred_params = {'in_features': d_model, 'out_features': type_vocab_size}

    lm = TypeFactoredLM(masked_encoder=Encoder,
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
