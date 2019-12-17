from TypeLM.utils.imports import *


def type_accuracy(predictions: LongTensor, truth: LongTensor,
                  ignore_idx: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    correct_items = torch.ones_like(predictions)
    correct_items[predictions != truth] = 0
    correct_items[truth == ignore_idx] = 1

    correct_sentences = correct_items.prod(dim=1)
    num_correct_sentences = correct_sentences.sum().item()

    num_correct_items = correct_items.sum().item()
    num_masked_items = len(truth[truth == ignore_idx])

    return (predictions.shape[0], num_correct_sentences), \
           (predictions.shape[0] * predictions.shape[1] - num_masked_items, num_correct_items - num_masked_items)


def positional_encoding(b: int, n: int, d_model: int, freq: int = 10000, device: str = 'cpu',
                        dropout_rate: float = 0.1) -> Tensor:
    pe = torch.zeros(n, d_model, device=device)
    position = torch.arange(0, n, device=device, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device, dtype=torch.float) *
                         - (torch.log(torch.tensor(freq, dtype=torch.float, device=device)) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.repeat(b, 1, 1)

    return F.dropout(pe, dropout_rate)


def count_parameters(model: Module) -> int:
    total_param = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_param = torch.prod(torch.tensor(param.size()))
            if param.dim() > 1:
                print(name, ':', 'x'.join(str(x) for x in list(param.size())), '=', num_param)
            else:
                print(name, ':', num_param)
            total_param += num_param
    return total_param
