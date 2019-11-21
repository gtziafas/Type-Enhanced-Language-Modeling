from src.utils.imports import *
from src.model.multi_head_attention import MultiHeadAttention, PositionWiseFeedForward


def make_encoder(module_maker: Module, num_layers: int, *args, **kwargs) -> Module:
    layers = [module_maker(*args, **kwargs) for _ in range(num_layers)]
    return Sequential(*layers)


class EncoderLayer(Module):
    def __init__(self, num_heads: int, d_k: int, d_model: int, d_v: int, d_ff: int = 2048, dropout_rate: float = 0.1) -> None:
        super(EncoderLayer, self).__init__()
        self.position_wise = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff, activation_fn=F.gelu)
        self.mha = MultiHeadAttention(num_heads=num_heads, d_k=d_k, d_model=d_model, d_v=d_v)
        self.layer_norm_1 = LayerNorm(d_model)
        self.layer_norm_2 = LayerNorm(d_model)
        self.dropout_rate = dropout_rate

    def forward(self, x: Tuple[FloatTensor, LongTensor]) -> Tuple[FloatTensor, LongTensor]:
        inputs, mask = x[0], x[1]
        attended = self.mha(inputs, inputs, inputs, mask)
        attended = attended + inputs
        attended_norm = self.layer_norm_1(attended)
        attended_norm = F.dropout(attended_norm, self.dropout_rate)

        transformed = self.position_wise(attended_norm)
        transformed = attended_norm + transformed
        transformed_norm = self.layer_norm_2(transformed)
        transformed_norm = F.dropout(transformed_norm, self.dropout_rate)

        return transformed_norm, mask


class MultiOutputEncoder(Module):
    def __init__(self, module_maker: Module, bottom_depth: int, top_depth: int, *args, **kwargs) -> None:
        super(MultiOutputEncoder, self).__init__()
        self.bottom = make_encoder(module_maker, bottom_depth, *args, **kwargs)
        self.top = make_encoder(module_maker, top_depth, *args, **kwargs)

    def forward(self, x: Tuple[FloatTensor, LongTensor]) -> Tuple[Tuple[FloatTensor, LongTensor],
                                                                  Tuple[FloatTensor, LongTensor]]:
        intermediate = self.bottom(x)
        final = self.top(intermediate)
        return intermediate, final








