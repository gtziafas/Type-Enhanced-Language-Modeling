from src.utils.imports import *
from src.model.multi_head_attention import MultiHeadAttention
from torch.nn import LayerNorm


def MaskedEncoder(module_maker: Module, num_layers: int, *args, **kwargs) -> Module:
    return Sequential(*[module_maker(*args, **kwargs) for _ in range(num_layers)])


class MaskedEncoderLayer(Module):
    def __init__(self, position_wise: Module, num_heads: int, d_k: int, d_model: int, d_v: int) -> None:
        super(MaskedEncoderLayer, self).__init__()
        self.position_wise = position_wise
        self.mha = MultiHeadAttention(num_heads, d_k, d_model, d_v)
        self.layer_norm_1 = LayerNorm(d_model)
        self.layer_norm_2 = LayerNorm(d_model)

    def forward(self, inputs: FloatTensor, mask: LongTensor, dropout_rate: float = 0.0) -> FloatTensor:
        attended = self.mha(inputs, inputs, inputs, mask)
        attended = attended + inputs
        attended_norm = self.layer_norm_1(attended)
        attended_norm = F.dropout(attended_norm, dropout_rate)

        transformed = self.position_wise(attended_norm)
        transformed = attended_norm + transformed
        transformed_norm = self.layer_norm_2(transformed)
        transformed_norm = F.dropout(transformed_norm, dropout_rate)
        return transformed_norm






