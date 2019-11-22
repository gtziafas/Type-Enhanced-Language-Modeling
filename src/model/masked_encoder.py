from src.utils.imports import *
from src.model.multi_head_attention import MultiHeadAttention, PositionWiseFeedForward


class EncoderLayer(Module):
    def __init__(self, num_heads: int, d_k: int, d_model: int, d_v: int, activation_fn: tensor_map, d_ff: int = 2048,
                 dropout_rate: float = 0.1) -> None:
        super(EncoderLayer, self).__init__()
        self.position_wise = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff, activation_fn=activation_fn)
        self.mha = MultiHeadAttention(num_heads=num_heads, d_k=d_k, d_model=d_model, d_v=d_v)
        self.layer_norm_1 = LayerNorm(d_model)
        self.layer_norm_2 = LayerNorm(d_model)
        self.dropout_rate = dropout_rate

    def forward(self, x: Tuple[FloatTensor, LongTensor]) -> Tuple[Tensor, LongTensor]:
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


class Encoder(Module):
    def __init__(self, module_maker: Module, num_layers: int, *args, **kwargs) -> None:
        super(Encoder, self).__init__()
        self.network = Sequential(*[module_maker(*args, **kwargs) for _ in range(num_layers)])

    def forward(self, x: FloatTensor, mask: LongTensor) -> Tensor:
        return self.network((x, mask))[0]


class WeightedLayerEncoder(Module):
    def __init__(self, module_maker: Module, num_layers: int, *args, **kwargs) -> None:
        super(WeightedLayerEncoder, self).__init__()
        self.layers = ModuleList([module_maker(*args, **kwargs) for _ in range(num_layers)])
        self.layer_weights = torch.nn.Parameter(torch.rand(num_layers), requires_grad=True)

    def forward(self, x: FloatTensor, mask: LongTensor) -> Tuple[Tensor, Tensor]:
        xs = [x]
        for layer in self.layers:
            xs = xs + [layer((xs[-1], mask))[0]]
        stacked = torch.stack(xs[1::])
        layer_weights = F.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
        return (layer_weights * stacked).sum(dim=0), xs[-1]