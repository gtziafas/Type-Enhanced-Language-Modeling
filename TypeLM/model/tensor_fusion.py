from TypeLM.utils.imports import * 
from torch.nn import Conv2d, MaxPool2d, GELU


class Outter2dFusion(Module):
    def __init__(self) -> None:
        super(Outter2dFusion, self).__init__()

    @staticmethod
    def outter2d_fn(x: Tensor, y: Tensor) -> Tensor:
        org_shape, d_model = x.shape[:-1], x.shape[-1]
        num_samples = x.view(-1, d_model).shape[0]
        xs = [x.view(-1, d_model)[s, :] for s in range(num_samples)]
        ys = [y.view(-1, d_model)[s, :] for s in range(num_samples)]
        gers = [torch.ger(xs[s], ys[s]) for s in range(num_samples)]
        return torch.stack(gers, dim=0).view(*org_shape, d_model, d_model).contiguous()

    def forward(self, gate: Tensor, features: Tensor) -> Tensor:
        return self.outter2d_fn(gate, features)
        

class ConvFeatures(Module):
    def __init__(self, depth: int=3, start_kernel: int=50, start_stride: int=3) -> None:
        super(ConvFeatures, self).__init__()
        assert depth > 2, 'Must have at least one intermediate conv block'
        blocks = [self.conv_block(in_channels=1, out_channels=16, conv_kernel=start_kernel, conv_stride=start_stride)]
        blocks[1:] = [self.conv_block(in_channels=16*(d+1), out_channels=16*(d+2)) for d in range(0, depth-2)]
        blocks.append(self.conv_block(in_channels=16*(depth-1), out_channels=16*(depth-1)))
        
        self.features = Sequential(*blocks)

    def conv_block(self, in_channels: int, out_channels: int, conv_kernel: int=3, conv_stride: int=1, pool_kernel: int=3) -> Module:
        return Sequential(Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=conv_kernel, stride=conv_stride),
                          GELU(),
                          MaxPool2d(kernel_size=pool_kernel))

    def forward(self, x: Tensor) -> Tensor:
        return self.features(x)


class Conv2dFusion(Module):
    def __init__(self, fusion: Outter2dFusion, conv: ConvFeatures, fusion_kwargs: Dict, conv_kwargs: Dict, dropout_rate: float=0.1) -> None:
        super(Conv2dFusion, self).__init__()
        self.fusion = fusion(**fusion_kwargs)
        self.conv = conv(**conv_kwargs)
        self.dropout = Dropout(p=dropout_rate)
    
    def forward(self, token_features: Tensor, type_preds: Tensor) -> Tensor:
        batch_size, seq_len, d_model = type_preds.shape

        fusion = self.fusion(gate=type_preds, features=token_features) # [B S D] x [B S D] -> [B S D D]
        convolved = [self.conv(fusion[:,w,:].unsqueeze(1)) for w in range(seq_len)] # a list of S [B 576] tensors
        convolved = self.dropout(torch.stack(convolved, dim=1).contiguous()) # [B S 576]

        return convolved.view(batch_size, seq_len, -1)