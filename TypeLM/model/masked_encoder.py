from TypeLM.utils.imports import *
from TypeLM.model.multi_head_attention import MultiHeadAttention, PositionWiseFeedForward


class EncoderLayer(Module):
    def __init__(self, num_heads: int, d_k: int, d_model: int, d_v: int, activation_fn: tensor_map, d_ff: int = 2048,
                 dropout_rate: float = 0.1) -> None:
        super(EncoderLayer, self).__init__()
        self.position_wise = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff, activation_fn=activation_fn)
        self.mha = MultiHeadAttention(num_heads=num_heads, d_k=d_k, d_model=d_model, d_v=d_v, dropout_rate=dropout_rate)
        self.layer_norm_1 = LayerNorm(d_model)
        self.layer_norm_2 = LayerNorm(d_model)
        self.dropout = Dropout(dropout_rate)

    def forward(self, x: Tuple[Tensor, LongTensor]) -> Tuple[Tensor, LongTensor]:
        inputs, mask = x[0], x[1]
        attended = self.mha(inputs, inputs, inputs, mask)
        attended = attended + inputs
        attended_norm = self.layer_norm_1(attended)
        attended_norm = self.dropout(attended_norm)

        transformed = self.position_wise(attended_norm)
        transformed = attended_norm + transformed
        transformed_norm = self.layer_norm_2(transformed)
        transformed_norm = self.dropout(transformed_norm)

        return transformed_norm, mask


class Encoder(Module):
    def __init__(self, module_maker: Module, num_layers: int, *args, **kwargs) -> None:
        super(Encoder, self).__init__()
        self.layers = ModuleList([module_maker(*args, **kwargs) for _ in range(num_layers)])

    def forward(self, x: Tensor, mask: LongTensor) -> Tensor:
        for layer in self.layers:
            x = layer((x, mask))[0]
        return x

    def forward_all(self, x: Tensor, mask: LongTensor) -> Tensor:
        xs = [x]
        for layer in self.layers:
            xs = xs + [layer((xs[-1], mask))[0]]
        return torch.stack(xs)


class LayerWeighter(Module):
    def __init__(self, num_layers: int) -> None:
        super(LayerWeighter, self).__init__()
        self.layer_weights = torch.nn.Parameter(torch.rand(num_layers), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        atn = F.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
        return (atn * x).sum(dim=0)
