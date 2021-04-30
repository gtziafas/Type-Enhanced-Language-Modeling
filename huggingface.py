from TypeLM.neural.defaults import default_model, TypedLM, Tokenizer
from TypeLM.neural.embedding import positional_encoding
from transformers import BertConfig, BertModel

from torch import Tensor
from typing import Tuple
from torch.nn import Linear, Module
from torch.nn.functional import linear
from math import sqrt


def make_config(model: TypedLM) -> BertConfig:
    return BertConfig(vocab_size=model.word_embedder.num_embeddings,
                      hidden_dim=model.word_embedder.embedding_dim,
                      num_hidden_layers=len(model.encoder_1) + len(model.encoder_2),
                      num_attention_heads=model.encoder_1[0].mha.num_heads,
                      intermediate_size=model.encoder_1[0].ffn.linear_one.out_features,
                      hidden_act='gelu',
                      hidden_dropout_prob=model.encoder_1[0].ffn.dropout.p,
                      attention_probs_dropout_prob=model.encoder_1[0].mha.dropout.p,
                      max_position_embeddings=100,
                      type_vocab_size=1,
                      layer_norm_eps=model.encoder_1[0].ln_mha.eps,
                      position_embedding_type='absolute',
                      pad_token_id=model.word_embedder.padding_idx,
                      output_hidden_states=True)


def make_bertmodel(source: TypedLM) -> BertModel:
    cfg = make_config(source)
    tgt = BertModel(cfg)
    tgt.embeddings.word_embeddings.weight.data = \
        source.word_embedder.embedding_matrix.data * source.word_embedder.embedding_scale
    tgt.embeddings.position_embeddings.weight.data = \
        positional_encoding(1, cfg.max_position_embeddings, cfg.hidden_size).squeeze(0)
    tgt.embeddings.token_type_embeddings.weight.data.fill_(0.)
    for i in range(cfg.num_hidden_layers):
        match = source.encoder_1[i] if i < len(source.encoder_1) else source.encoder_2[i - len(source.encoder_1)]
        tgt_layer = tgt.encoder.layer[i]

        tgt_layer.attention.self.query.weight.data, tgt_layer.attention.self.query.bias.data = \
            match.mha.q_transformation.weight.data, match.mha.q_transformation.bias.data
        tgt_layer.attention.self.key.weight.data, tgt_layer.attention.self.key.bias.data = \
            match.mha.k_transformation.weight.data, match.mha.k_transformation.bias.data
        tgt_layer.attention.self.value.weight.data, tgt_layer.attention.self.value.bias.data = \
            match.mha.v_transformation.weight.data, match.mha.v_transformation.bias.data
        tgt_layer.attention.output.dense.weight.data, tgt_layer.output.dense.bias.data = \
            match.mha.wo.weight.data, match.mha.wo.bias.data
        tgt_layer.attention.output.LayerNorm.weight.data, tgt_layer.attention.output.LayerNorm.bias.data = \
            match.ln_mha.weight.data, match.ln_mha.bias.data
        tgt_layer.intermediate.dense.weight.data, tgt_layer.intermediate.dense.bias.data = \
            match.ffn.linear_one.weight.data, match.ffn.linear_one.bias.data
        tgt_layer.output.dense.weight.data, tgt_layer.output.dense.bias.data = \
            match.ffn.linear_two.weight.data, match.ffn.linear_two.bias.data
        tgt_layer.output.dense.weight.data, tgt_layer.output.dense.bias.data = \
            match.ffn.linear_two.weight.data, match.ffn.linear_two.bias.data
        tgt_layer.output.LayerNorm.weight.data, tgt_layer.output.LayerNorm.bias.data = \
            match.ln_ffn.weight.data, match.ln_ffn.bias.data
    return tgt


class Wrapped(Module):
    def __init__(self, bert: BertModel, type_cls: Linear, tokenizer: Tokenizer):
        super(Wrapped, self).__init__()
        self.bert = bert
        self.type_cls = type_cls
        self.tokenizer = tokenizer

    def forward_train(self, xs: Tensor, mask: Tensor):
        hs = self.bert(xs, mask)['hidden_states']
        types_out = self.type_cls(hs[5])
        words_out = linear(hs[-1], self.bert.embeddings.word_embeddings.weight / sqrt(self.bert.config.hidden_size))
        return words_out, types_out

    @staticmethod
    def from_typedlm(typed_lm: TypedLM) -> 'Wrapped':
        bert = make_bertmodel(typed_lm)
        return Wrapped(bert, typed_lm.type_classifier, typed_lm.tokenizer)
