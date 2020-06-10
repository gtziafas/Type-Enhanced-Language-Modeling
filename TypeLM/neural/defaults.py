from TypeLM.neural.model import TypedLM
from TypeLM.neural.loss import *
from TypeLM.neural.optimizer import Scheduler, make_noam_scheme
from torch.nn import CrossEntropyLoss
from TypeLM.preprocessing.defaults import *
from torch.optim import AdamW

tokenizer = default_tokenizer()


def default_model() -> TypedLM:
    return TypedLM(tokenizer, 768, 768, (6, 12), 3, 12, 12, 'cuda')


def default_loss() -> MixedLoss:
    mlm_loss_kwargs = {'ignore_index': tokenizer.word_tokenizer.core.pad_token_id,
                       'reduction': 'mean'}
    st_loss_kwargs = {'num_classes': len(tokenizer.type_tokenizer.vocabulary),
                      'mass_redistribution': 0.1,
                      'ignore_index': [tokenizer.type_tokenizer.PAD_TOKEN_ID],
                      'reduction': 'batchmean'}
    return MixedLoss(CrossEntropyLoss, FuzzyLoss, mlm_loss_kwargs, st_loss_kwargs, 1)


def default_optimizer(model: TypedLM, warmup_steps: int) -> Scheduler:
    schedule = make_noam_scheme(d_model=768, warmup_steps=warmup_steps, factor=1.)
    _opt = AdamW(model.parameters(), lr=1e10, betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-05)
    return Scheduler(_opt, schedule, [1])
