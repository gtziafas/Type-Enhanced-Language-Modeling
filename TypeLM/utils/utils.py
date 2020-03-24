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


def positional_encoding(b: int, n: int, d_model: int, freq: int = 10000, device: str = 'cpu') -> Tensor:
    pe = torch.zeros(n, d_model, device=device)
    position = torch.arange(0, n, device=device, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device, dtype=torch.float) *
                         - (torch.log(torch.tensor(freq, dtype=torch.float, device=device)) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.repeat(b, 1, 1)


def sigsoftmax(x: Tensor, dim: int) -> Tensor:
    sigx = torch.sigmoid(x) * torch.exp(x)
    rank = len(sigx.shape)
    norm = torch.sum(sigx, dim=dim).unsqueeze(dim).repeat([1 if i != dim else sigx.shape[i] for i in range(rank)])
    return sigx/norm


class PositionalEncoder(Module):
    def __init__(self, dropout_rate: float = 0.1):
        super(PositionalEncoder, self).__init__()
        self.dropout = Dropout(p=dropout_rate)

    def forward(self, b: int, n: int, d_model: int, freq: int = 1000, device: str = 'cpu'):
        return self.dropout(positional_encoding(b, n, d_model, freq, device))


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


class CustomLRScheduler(object):
    def __init__(self, optimizer: Optimizer, update_fns: Sequence[Callable[[int, Any], float]],
                 **kwargs: Any) -> None:
        assert len(update_fns) == len(optimizer.param_groups)
        self.opt = optimizer
        self._step = 0
        self.update_fns = update_fns
        self.lrs = [None for _ in range(len(self.opt.param_groups))]
        self.__dict__.update(kwargs)

    def step(self) -> None:
        self._step += 1
        self.lrs = self.update(step=self._step, **{k: v for k, v in self.__dict__.items() if k not in
                                                   ('_step', 'opt', 'update_fns', 'lrs')})
        for i, p in enumerate(self.opt.param_groups):
            p['lr'] = self.lrs[i]
        self.opt.step()

    def zero_grad(self) -> None:
        self.opt.zero_grad()

    def update(self, step: int, **kwargs) -> List[float]:
        return [update_fn(step, **kwargs) for update_fn in self.update_fns]


def noam_scheme(_step: int, d_model: int, warmup_steps: int, batch_size: int=2048) -> float:
    return d_model**-0.5 * min(_step**-0.5, _step*warmup_steps**-1.5) * batch_size/2048


def linear_scheme(_step: int, warmup_steps: int, goal_lr: float, decrease_rate: int, min_lr: float) -> float:
    def threshold(x):
        return x if x > min_lr else min_lr

    return goal_lr * _step/warmup_steps if _step < warmup_steps \
        else threshold(goal_lr - decrease_rate * (_step - warmup_steps))


def save_model(model: Module, save_id: str,
               opt: Optimizer, num_epochs: int, loss: Tuple[float, float],
               data_dir="./TypeLM/checkpoints/",
               ) -> None:
    # create dir if not there 
    import os 
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    save_to = data_dir + str(save_id) + '.pth'
    torch.save({
        'epoch'                 :   num_epochs,
        'loss'                  :   loss,
        'model_state_dict'      :   model.state_dict(),
        'optimizer_state_dict'  :   opt.state_dict()
    }, save_to)


class ElementWiseFusion(Module):
    def __init__(self, activation: tensor_map = F.tanh):
        super(ElementWiseFusion, self).__init__()
        self.activation = activation

    def forward(self, gate: Tensor, features: Tensor) -> Tensor:
        return self.activation(gate) * features


def load_model(model_path: str, model: Module, opt: Optimizer) -> Tuple[Module, Optimizer, int, Tuple[float, float]]:
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    num_epochs = checkpoint['epoch'] + 1
    loss = checkpoint['loss'] 
    return model, opt, num_epochs, loss


def label_smoothing(x: LongTensor, num_classes: int, smoothing: float, ignore_index: Optional[int] = None) -> Tensor:
    x_float = torch.ones(x.shape, device=x.device, dtype=torch.float).unsqueeze(-1)
    x_float = x_float.repeat([1 for _ in x.shape] + [num_classes])
    x_float.fill_(smoothing / (num_classes - 1))
    x_float.scatter_(dim=-1, index=x.unsqueeze(-1), value=1-smoothing)
    if ignore_index is not None:
        mask = x == ignore_index
        x_float[mask.unsqueeze(-1).repeat([1 for _ in x.shape] + [num_classes])] = 0
    return x_float


class LabelSmoother(Module):
    def __init__(self, num_classes: int, smoothing: float, ignore_index: Optional[int] = None):
        super(LabelSmoother, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def forward(self, x: LongTensor, ignore_index: Optional[int] = None,
                smoothing: Optional[float] = None) -> Tensor:
        return label_smoothing(x, self.num_classes,
                               smoothing if smoothing is not None else self.smoothing,
                               ignore_index if ignore_index is not None else self.ignore_index)


class FuzzyLoss(Module):
    def __init__(self, num_classes: int, mass_redistribution: float, ignore_index: int = 0) -> None:
        super(FuzzyLoss, self).__init__()
        self.loss_fn = KLDivLoss(reduction='batchmean')
        self.label_smoother = LabelSmoother(num_classes, mass_redistribution, ignore_index)

    def __call__(self, x: Tensor, y: LongTensor) -> Tensor:
        smooth_y = self.label_smoother(y)
        return self.loss_fn(torch.log_softmax(x, dim=-1), smooth_y)


class CrossEntropySS(Module):
    def __init__(self, **kwargs):
        super(CrossEntropySS, self).__init__()
        self.NLL = torch.nn.NLLLoss(**kwargs)

    def forward(self, predictions: Tensor, truth: LongTensor) -> Tensor:
        predictions = sigsoftmax(predictions, dim=-1)
        return self.NLL(predictions.log(), truth)