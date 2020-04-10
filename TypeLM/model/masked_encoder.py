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
    
    @overload
    def forward(self, x: Tuple[Tensor, Tensor, Tensor, LongTensor]) -> Tuple[Tensor, LongTensor]:
        pass
    
    @overload
    def forward(self, x: Tuple[Tensor, LongTensor]) -> Tuple[Tensor, LongTensor]:
        pass
    
    def forward(self, x):
        if len(x) == 2:
            inputs, mask = x
            return self.transform((inputs, inputs, inputs, mask))
        elif len(x) == 4:
            return self.transform(x)
        else:
            raise TypeError('Expected a tuple of either 2 or 4 tensors')

    def transform(self, x: Tuple[Tensor, Tensor, Tensor, LongTensor]) -> Tuple[Tensor, LongTensor]:
        queries, keys, values, mask = x
        attended = self.mha(queries, keys, values, mask)
        attended = attended + values
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
        
    @overload 
    def forward(self, x: Tensor, mask: LongTensor) -> Tensor:
        pass
    
    @overload
    def forward(self, queries: Tensor, keys: Tensor, values: Tensor, mask: LongTensor) -> Tensor:
        pass

    def forward(self, *args):
        if len(args) == 2:
            return self.forward_single(*args)
        elif len(args) == 4:
            return self.forward_many(*args)
        else:
            raise TypeError('Expected either 2 tensors for self or 4 for multi-head attention')

    def forward_single(self, x: Tensor, mask: LongTensor) -> Tensor:
        for layer in self.layers:
            x = layer((x, mask))[0]
        return x

    def forward_many(self, queries: Tensor, keys: Tensor, values: Tensor, mask: LongTensor) -> Tensor:
        x = (queries, keys, values, mask)
        for layer in self.layers:
            x = layer(x)
        return x[0]
    
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
