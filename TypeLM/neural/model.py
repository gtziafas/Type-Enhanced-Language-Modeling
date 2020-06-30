from TypeLM.preprocessing.tokenizer import Tokenizer
from TypeLM.neural.embedding import InvertibleEmbedder
from TypeLM.neural.transformer import make_encoder

from torch.nn import Module, Linear
from typing import Tuple, NoReturn

from torch import Tensor, LongTensor


class TypedLM(Module):
    def __init__(self, tokenizer: Tokenizer, encoder_dim: int, encoder_layers: Tuple[int, int],
                 encoder_heads: int, device: str):
        super(TypedLM, self).__init__()
        word_vocab_size = tokenizer.word_tokenizer.core.vocab_size
        word_padding_idx = tokenizer.word_tokenizer.core.pad_token_id
        self.word_embedder = InvertibleEmbedder(embedding_dim=encoder_dim, num_embeddings=word_vocab_size,
                                                padding_idx=word_padding_idx, scale_by_sqrt=True).to(device)
        type_vocab_size = len(tokenizer.type_tokenizer.vocabulary)
        self.type_classifier = Linear(in_features=encoder_dim, out_features=type_vocab_size).to(device)
        self.encoder_1 = make_encoder(encoder_layers[0], encoder_heads, encoder_dim, encoder_dim//encoder_heads,
                                      encoder_dim//encoder_heads, 2 * encoder_dim, 0.1).to(device)
        self.encoder_2 = make_encoder(encoder_layers[1], encoder_heads, encoder_dim, encoder_dim//encoder_heads,
                                      encoder_dim//encoder_heads, 2 * encoder_dim, 0.1).to(device)
        self.tokenizer = tokenizer
        self.device = device

    def forward(self, *args, **kwargs) -> NoReturn:
        raise NotImplementedError('Forward not implemented for this model.')

    def forward_train(self, word_ids: LongTensor, word_mask: LongTensor) -> Tuple[Tensor, Tensor]:
        shallow = self.encode_shallow(word_ids, word_mask)
        deep = self.encode_deep(shallow, word_mask)
        word_out = self.word_embedder.invert(deep)
        type_out = self.type_classifier(shallow)
        return word_out, type_out

    def encode_shallow(self, word_ids: LongTensor, word_mask: LongTensor) -> Tensor:
        word_reprs = self.word_embedder.embed(word_ids)
        return self.encoder_1((word_reprs, word_mask))[0]

    def encode_deep(self, shallow: Tensor, word_mask: LongTensor) -> Tensor:
        return self.encoder_2((shallow, word_mask))[0]
