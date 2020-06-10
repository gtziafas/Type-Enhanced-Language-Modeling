from typing import Callable, Sequence
from torch.optim import Optimizer


def make_noam_scheme(d_model: int, warmup_steps: int, factor: float) -> Callable[[int], float]:
    def noam_scheme(step: int) -> float:
        step += 1
        return d_model**-0.5 * min(step**-0.5, step*warmup_steps**-1.5) * factor
    return noam_scheme


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
