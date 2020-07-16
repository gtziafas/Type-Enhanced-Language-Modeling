from typing import Callable, Sequence
from torch.optim import Optimizer


def make_linear_schedule(warmup_steps: int, total_steps: int, max_lr: float) -> Callable[[int], float]:
    def linear_schedule(step: int) -> float:
        l = max_lr / (warmup_steps - total_steps)
        beta = total_steps * max_lr / (total_steps - warmup_steps)
        if step <= warmup_steps:
            return max_lr * step/warmup_steps
        return l * step + beta
    return linear_schedule


class Scheduler:
    def __init__(self, opt: Optimizer, schedule: Callable[[int], float], param_group_scales: Sequence[float] = (1,)):
        self.opt = opt
        self.schedule = schedule
        self.step_num = 0
        self.lr = 0
        self.param_group_scales = param_group_scales

    def step(self) -> None:
        self.step_num += 1
        self.lr = self.schedule(self.step_num)
        for i, p in enumerate(self.opt.param_groups):
            p['lr'] = self.lr * self.param_group_scales[i]
        self.opt.step()

    def zero_grad(self) -> None:
        self.opt.zero_grad()
