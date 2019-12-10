EOS = '[EOS]'       # End of Sentence Input
MASK = '[MASK]'     # Masked Input Word
NUM = '[NUM]'       # Numeral Input Word
PROC = '[PROC]'     # Input word truncated by preprocessing
UNK = '[UNK]'       # Input word truncated by frequency threshold
PAD = '[PAD]'       # Unknown Input Type
MWU = '[MWU]'       # Untyped MWU expression

word_input_tokens = {EOS, MASK, NUM, PROC, UNK}
type_input_tokens = {MWU}
