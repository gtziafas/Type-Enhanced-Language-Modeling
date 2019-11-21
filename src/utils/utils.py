from src.utils.imports import *


def positional_encoding(b: int, n: int, d_model: int, freq: int = 10000, device: str= 'cpu',
                        dropout_rate: float=0.1) -> FloatTensor:
    pe = torch.zeros(n, d_model, device=device)
    position = torch.arange(0, n, device=device, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device, dtype=torch.float) *
                         - (torch.log(torch.tensor(freq, dtype=torch.float, device=device)) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.repeat(b, 1, 1)

    return F.dropout(pe, dropout_rate)

