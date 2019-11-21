from src.utils.imports import *


def MultiHeadAttentionFn(queries: FloatTensor, keys: FloatTensor, values: FloatTensor,
                         qts: tensor_maps, kts: tensor_maps, vts: tensor_maps, wo: tensor_map,
                         mask: Optional[LongTensor] = None, dropout_rate: float = 0.1) -> FloatTensor:
    qs = [qt(queries) for qt in qts]
    ks = [kt(keys) for kt in kts]
    vs = [vt(values) for vt in vts]
    outputs = [ScaledDotProduct(qs[i], ks[i], vs[i], mask) for i in range(len(qs))]
    outputs = F.dropout(torch.cat(outputs, dim=-1), dropout_rate)
    return wo(outputs)


def ScaledDotProduct(queries: FloatTensor, keys: FloatTensor, values: FloatTensor,
                     mask: Optional[LongTensor] = None) -> FloatTensor:
    dk = keys.shape[-1]
    dividend = torch.tensor(dk, device=queries.device, dtype=torch.float)

    weights = torch.bmm(queries, keys.transpose(2, 1)) / torch.sqrt(dividend)  # [B, M, N]
    if mask is not None:
        weights = weights.masked_fill_(mask == 0, value=-1e10)
    weights = F.softmax(weights, dim=-1)  # [B, M, N] -- each m thing attends a probability distribution over N things
    return torch.bmm(weights, values)


class MultiHeadAttention(Module):
    def __init__(self, num_heads: int, d_model: int, d_k: int, d_v: int) -> None:
        super(MultiHeadAttention, self).__init__()
        self.q_transformations = ModuleList([Linear(in_features=d_model, out_features=d_k, bias=False)
                                             for _ in range(num_heads)])
        self.k_transformations = ModuleList([Linear(in_features=d_model, out_features=d_k, bias=False)
                                             for _ in range(num_heads)])
        self.v_transformations = ModuleList([Linear(in_features=d_model, out_features=d_v, bias=False)
                                             for _ in range(num_heads)])
        self.Wo = Linear(in_features=num_heads * d_v, out_features=d_model, bias=False)

    def forward(self, queries: FloatTensor, keys: FloatTensor, values: FloatTensor,
                mask: Optional[LongTensor] = None) -> FloatTensor:
        return MultiHeadAttentionFn(queries, keys, values, self.q_transformations, self.k_transformations,
                                    self.v_transformations, self.Wo, mask)


class PositionWiseFeedForward(Module):
    def __init__(self, d_model: int, activation_fn: tensor_map, d_ff: int, dropout_rate: float = 0.1) -> None:
        super(PositionWiseFeedForward, self).__init__()
        self.w_1 = Linear(d_model, d_ff)
        self.w_2 = Linear(d_ff, d_model)
        self.activation_fn = activation_fn
        self.dropout_rate = dropout_rate

    def forward(self, x):
        w1 = self.w_1(x)
        w1 = F.dropout(self.activation_fn(w1), self.dropout_rate)
        
        return self.w_2(w1)
