from TypeLM.neural.model import TypedLM
from TypeLM.neural.loss import *
from TypeLM.neural.optimizer import Scheduler, make_linear_schedule
from TypeLM.preprocessing.defaults import *
from torch.optim import AdamW

tokenizer = default_tokenizer()

_d_model = 768


def default_model() -> TypedLM:
    return TypedLM(tokenizer, _d_model, (4, 8), 12, 'cuda')


def default_loss() -> MixedLoss:
    mlm_loss_kwargs = {'ignore_index': [tokenizer.word_tokenizer.core.pad_token_id,
                                        tokenizer.word_tokenizer.core.cls_token_id,
                                        tokenizer.word_tokenizer.core.sep_token_id,
                                        tokenizer.word_tokenizer.core.unk_token_id,
                                        tokenizer.word_tokenizer.core.mask_token_id],
                       'reduction': 'mean'}
    st_loss_kwargs = {'ignore_index': [tokenizer.type_tokenizer.PAD_TOKEN_ID,
                                       tokenizer.type_tokenizer.UNK_TOKEN_ID],
                      'reduction': 'mean'}
    return MixedLoss(CrossEntropyLossMultiIgnore, CrossEntropyLossMultiIgnore, mlm_loss_kwargs, st_loss_kwargs, 1)

