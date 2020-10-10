from TypeLM.neural.defaults import *
from TypeLM.preprocessing.defaults import *

import torch

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model_path = '/data/s3913171/Lassy-Large/256_bertbase_1.pth'

tokenizer = default_tokenizer().word_tokenizer


def load_model(modelpath: str) -> TypedLM:
  model = default_model().to(_device)  
  tmp = torch.load(modelpath)
  model.load_state_dict(tmp['model_state_dict'])
  return model


def infer_sent(sent: str, 
        mask_idces: List[int],
        model: TypedLM,
        mask_token: int = 4,
        kappa: int = 20) -> List[List[str]]:  
  word_ids = tokenizer.convert_sent_to_ids(sent)
  word_ids = torch.tensor(word_ids, dtype=torch.long, device=_device)
  pad_mask = torch.ones(len(word_ids), len(word_ids), dtype=torch.long, device=_device)
  word_ids[mask_idces] = mask_token
  predictions = model.forward_train(word_ids.unsqueeze(0), pad_mask.unsqueeze(0))[0].squeeze()
  predictions =  predictions[mask_idces].topk(kappa)[1].tolist()
  return list(map(tokenizer.convert_ids_to_tokens, predictions))

def from_file():
  pass


def online():
  pass