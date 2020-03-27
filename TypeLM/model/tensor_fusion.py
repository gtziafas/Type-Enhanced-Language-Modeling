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
        

class Conv2dFeatures(Module):
    def __init__(self, depth: int, num_channels: int, start_kernel: int, start_stride: int, pool_kernel: int=3) -> None:
        super(Conv2dFeatures, self).__init__()
        blocks = [self.conv_block(in_channels=1, out_channels=num_channels, 
                  conv_kernel=start_kernel, conv_stride=start_stride, pool_kernel=pool_kernel)]
        blocks[1:] = [self.conv_block(in_channels=16*(d+1), out_channels=16*(d+2), pool_kernel=pool_kernel) for d in range(0, depth-2)]
        if depth > 1:
            blocks.append(self.conv_block(in_channels=16*(depth-1), out_channels=16*(depth-1), pool_kernel=pool_kernel))
        
        self.features = Sequential(*blocks)

    def conv_block(self, in_channels: int, out_channels: int, pool_kernel: int, conv_kernel: int=3, conv_stride: int=1, ) -> Module:
        return Sequential(Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=conv_kernel, stride=conv_stride),
                          GELU(),
                          MaxPool2d(kernel_size=pool_kernel))

    def forward(self, x: Tensor) -> Tensor:
        return self.features(x)


class Conv2dFusion(Module):
    def __init__(self, fusion: Outter2dFusion, conv: Conv2dFeatures, fusion_kwargs: Dict, conv_kwargs: Dict, dropout_rate: float=0.1) -> None:
        super(Conv2dFusion, self).__init__()
        self.fusion = fusion(**fusion_kwargs)
        self.conv = conv(**conv_kwargs)
        self.dropout = Dropout(p=dropout_rate)
    
    def forward(self, token_features: Tensor, type_embedds: Tensor) -> Tensor:
        batch_size, seq_len, d_model = type_embedds.shape

        fused = self.fusion(gate=type_embedds, features=token_features) # [B S D] x [B S D] -> [B S D D]
        convolved = [self.conv(fused[:,w,:].unsqueeze(1)) for w in range(seq_len)] # a list of S [B D] tensors7
        convolved = self.dropout(torch.stack(convolved, dim=1).contiguous()) # [B S D]

        return convolved.view(batch_size, seq_len, -1)


def example():
    batch_size = 2 
    seq_len = 3
    d_model = 512

    x = torch.rand(batch_size, seq_len, d_model)
    y = torch.rand_like(x) 

    deep_params = {'depth':3, 'num_channels':16, 'start_kernel':50, 'start_stride':3, 'pool_kernel':3}
    shallow_params = {'depth':1, 'num_channels':32, 'start_kernel':130, 'start_stride':11, 'pool_kernel':8}

    m1 = Conv2dFusion(fusion=Outter2dFusion, conv=Conv2dFeatures, fusion_kwargs={}, conv_kwargs=deep_params)
    m2 = Conv2dFusion(fusion=Outter2dFusion, conv=Conv2dFeatures, fusion_kwargs={}, conv_kwargs=shallow_params)

    print('with 3 blocks= {}'.format(m1(x,y).shape))
    print('with 1 block= {}'.format(m2(x,y).shape))                                                                                                                                                                                                                                                                                                      