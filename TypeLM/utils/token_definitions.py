# Lexical Input
EOS = '[EOS]'       # End of Sentence Input
MASK = '[MASK]'     # Masked Input Word
NUM = '[NUM]'       # Numeral Input Word
PROC = '[PROC]'     # Input word truncated by preprocessing
UNK = '[UNK]'       # Input word truncated by frequency threshold

# Lexical Output
PAD = '[PAD]'       # Padded output word

# Type Output
MWU = '[MWU]'       # Untyped MWU expression


lexical_input_tokens = {EOS, MASK, NUM, PROC, UNK}

tokens = {EOS, MASK, NUM, PROC, UNK, PAD, MWU}
