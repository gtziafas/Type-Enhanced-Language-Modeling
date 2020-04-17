from TypeLM.utils.imports import *
from TypeLM.utils.utils import PositionalEncoder, count_parameters
from TypeLM.model.masked_encoder import EncoderLayer, LayerWeighter, Encoder
from TypeLM.model.loss import LabelSmoother


class TypeFactoredLM(Module):
    def __init__(self, masked_encoder: Module, type_classifier: Module, type_embedder: Module,
                 fusion: Module, prefuse_encoder_kwargs: Dict, fused_encoder_kwargs: Dict,
                 num_words: int, type_classifier_kwargs: Dict, type_embedder_kwargs: Dict, 
                 fusion_kwargs: Dict, label_smoother_kwargs: Dict, padding_idx: int = 0, 
                 dropout_rate: float = 0.1) -> None:
        super(TypeFactoredLM, self).__init__()
        self.d_model = prefuse_encoder_kwargs['d_model']
        self.prefuse_encoder = masked_encoder(**prefuse_encoder_kwargs)
        self.fused_encoder = masked_encoder(**fused_encoder_kwargs)
        self.layer_weighter = LayerWeighter(prefuse_encoder_kwargs['num_layers'])
        self.type_classifier = type_classifier(**type_classifier_kwargs)
        self.type_embedder = type_embedder(**type_embedder_kwargs)
        self.fusion = fusion(**fusion_kwargs)
        self.word_embedder = Embedding(num_embeddings=num_words, embedding_dim=self.d_model,
                                       padding_idx=padding_idx)
        self.positional_encoder = PositionalEncoder(dropout_rate)
        self.dropout = Dropout(dropout_rate)
        self.label_smoother = LabelSmoother(**label_smoother_kwargs)

    def forward(self, word_ids: LongTensor, pad_mask: LongTensor,
                type_guidance: Optional[LongTensor] = None,
                confidence: float = 0.9,
                ignore_idx: int = -1,
                smoothing: Optional[float] = 0,
                embed_output: bool = True) -> Tuple[Tensor, Tensor]:
        layer_outputs = self.get_prefuse_vectors(word_ids, pad_mask)
        weighted = self.layer_weighter(layer_outputs[1:])
        type_preds = self.type_classifier(self.dropout(weighted))
        type_probs = type_preds.softmax(dim=-1)
        if type_guidance is not None:
            #guidance_indices = type_guidance != ignore_idx
            #smoothed_guidance = self.label_smoother(type_guidance[guidance_indices], smoothing) * (1 - confidence)
            #smoothed_guidance = smoothed_guidance + confidence * type_probs[guidance_indices]
            #type_probs[guidance_indices] = smoothed_guidance
            smoothed_guidance = self.label_smoother(type_guidance, smoothing) * (1 - confidence)
            type_probs = smoothed_guidance + confidence * type_probs
        type_embeddings = self.type_embedder(type_probs)
        token_features = self.fusion(type_embeddings, layer_outputs[-1])
        token_features = self.fused_encoder(token_features, pad_mask)

        ## alternatively for multi-head instead of self-attention as first step:
        ## token_features = self.fused_encoder(token_features, layer_outputs[-1], layer_outputs[-1], pad_mask)

        return self.word_classifier(token_features) if embed_output else token_features, type_preds

    def forward_st(self, word_ids: LongTensor, pad_mask: LongTensor) -> Tensor:
        layer_outputs = self.get_prefuse_vectors(word_ids, pad_mask)
        weighted = self.layer_weighter(layer_outputs[1:])
        type_preds = self.type_classifier(weighted)
        return type_preds

    def get_prefuse_vectors(self, word_ids: LongTensor, pad_mask: LongTensor) -> Tensor:
        # dividend = torch.sqrt(torch.tensor(self.d_model, dtype=torch.float, device=word_ids.device, requires_grad=False))
        word_embeds = self.word_embedder(word_ids) 
        batch_size, num_words, d_model = word_embeds.shape[0:3]
        positional_encodings = self.positional_encoder(b=batch_size, n=num_words, d_model=d_model,
                                                       device=word_embeds.device.__str__())

        word_embeds = word_embeds + positional_encodings
        return self.prefuse_encoder.forward_all(word_embeds, pad_mask)

    def word_classifier(self, final: Tensor) -> Tensor:
        return final@self.word_embedder.weight.transpose(1, 0)

    def get_token_features(self, word_ids: LongTensor, pad_mask: LongTensor) -> Tensor:
        return self.forward(word_ids, pad_mask, embed_output=False)[0]
