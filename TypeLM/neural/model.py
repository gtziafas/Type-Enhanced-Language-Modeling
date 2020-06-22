from TypeLM.preprocessing.tokenizer import Tokenizer
from TypeLM.neural.embedding import InvertibleEmbedder
from TypeLM.neural.transformer import make_encoder, make_decoder

from torch.nn import Module, Linear
from typing import Tuple, NoReturn

from torch import Tensor, LongTensor
import torch


class ClassifyingTypeLM(Module):
    def __init__(self, tokenizer: Tokenizer, encoder_dim: int, encoder_layers: Tuple[int, int],
                 encoder_heads: int, device: str):
        super(ClassifyingTypeLM, self).__init__()
        word_vocab_size = tokenizer.word_tokenizer.core.vocab_size
        word_padding_idx = tokenizer.word_tokenizer.core.pad_token_id
        self.word_embedder = InvertibleEmbedder(embedding_dim=encoder_dim, num_embeddings=word_vocab_size,
                                                padding_idx=word_padding_idx, scale_by_sqrt=True).to(device)
        type_vocab_size = len(tokenizer.type_tokenizer.vocabulary)
        self.type_classifier = Linear(in_features=encoder_dim, out_features=type_vocab_size).to(device)
        self.encoder_1 = make_encoder(encoder_layers[0], encoder_heads, encoder_dim, encoder_dim, encoder_dim,
                                      2 * encoder_dim, 0.1).to(device)
        self.encoder_2 = make_encoder(encoder_layers[1], encoder_heads, encoder_dim, encoder_dim, encoder_dim,
                                      2 * encoder_dim, 0.1).to(device)
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


class DecodingTypeLM(Module):
    def __init__(self, tokenizer: Tokenizer, encoder_dim: int, decoder_dim: int, encoder_layers: Tuple[int, int],
                 encoder_heads: int, decoder_layers: int, decoder_heads: int, device: str):
        super(DecodingTypeLM, self).__init__()
        word_vocab_size = tokenizer.word_tokenizer.core.vocab_size
        word_padding_idx = tokenizer.word_tokenizer.core.pad_token_id
        self.word_embedder = InvertibleEmbedder(embedding_dim=encoder_dim, num_embeddings=word_vocab_size,
                                                padding_idx=word_padding_idx, scale_by_sqrt=True).to(device)
        type_vocab_size = len(tokenizer.type_tokenizer.vocabulary)
        type_padding_idx = tokenizer.type_tokenizer.PAD_TOKEN_ID
        self.type_embedder = InvertibleEmbedder(embedding_dim=decoder_dim, num_embeddings=type_vocab_size,
                                                padding_idx=type_padding_idx, scale_by_sqrt=True).to(device)
        self.encoder_1 = make_encoder(encoder_layers[0], encoder_heads, encoder_dim, encoder_dim, encoder_dim,
                                      2 * encoder_dim, 0.1).to(device)
        self.encoder_2 = make_encoder(encoder_layers[1], encoder_heads, encoder_dim, encoder_dim, encoder_dim,
                                      2 * encoder_dim, 0.1).to(device)
        self.decoder = make_decoder(num_layers=decoder_layers, num_heads_enc=encoder_heads, num_heads_dec=decoder_heads,
                                    d_encoder=encoder_dim, d_decoder=decoder_dim, d_atn_enc=encoder_dim//encoder_heads,
                                    d_atn_dec=decoder_dim//decoder_heads, d_v_enc=encoder_dim//encoder_heads,
                                    d_v_dec=decoder_dim//decoder_heads, d_interm=decoder_dim*2,
                                    dropout_rate=0.1).to(device)
        self.tokenizer = tokenizer
        self.device = device

    def forward(self, *args, **kwargs) -> NoReturn:
        raise NotImplementedError('Forward not implemented for this model.')

    def forward_train(self, word_ids: LongTensor, word_mask: LongTensor, type_ids: LongTensor) -> Tuple[Tensor, Tensor]:
        shallow = self.encode_shallow(word_ids, word_mask)
        deep = self.encode_deep(shallow, word_mask)
        decode = self.decode_train(shallow, word_mask, type_ids)
        word_out = self.word_embedder.invert(deep)
        type_out = self.type_embedder.invert(decode)
        return word_out, type_out

    def encode_shallow(self, word_ids: LongTensor, word_mask: LongTensor) -> Tensor:
        word_reprs = self.word_embedder.embed(word_ids)
        return self.encoder_1((word_reprs, word_mask))[0]

    def encode_deep(self, shallow: Tensor, word_mask: LongTensor) -> Tensor:
        return self.encoder_2((shallow, word_mask))[0]

    def make_decoder_mask(self, b: int, n: int) -> LongTensor:
        upper_triangular = torch.triu(torch.ones(b, n, n), diagonal=1)
        return (torch.ones(b, n, n) - upper_triangular).to(self.device)

    def expand_encoder_mask(self, encoder_mask: LongTensor, n: int) -> LongTensor:
        return encoder_mask[:, 0, :].unsqueeze(1).repeat(1, n, 1)

    def decode_train(self, shallow: Tensor, word_mask: LongTensor, type_ids: LongTensor) -> Tensor:
        type_reprs = self.type_embedder.embed(type_ids)
        decoder_mask = self.make_decoder_mask(type_ids.shape[0], type_ids.shape[1])
        word_mask = self.expand_encoder_mask(word_mask, type_ids.shape[1])
        return self.decoder((shallow, word_mask, type_reprs, decoder_mask))[2]

