from TypeLM.model.train import *
from TypeLM.model.masked_encoder import EncoderLayer, Encoder
import sys

def default_dataloader(path: str = '/data/s3913171/Lassy-Large/out.txt', chunk_size = 10240, batch_size=128) -> DataLoader:
    masker = default_masker()

    def post_processor(sentences: Samples) -> Tuple[LongTensor, LongTensor, LongTensor, LongTensor, LongTensor]:
        true_words, types = list(zip(*sentences))
        masked_words, masked_indices = list(zip(*list(map(masker, true_words))))
        masked_words = pad_sequence(list(map(LongTensor, masked_words)))
        true_words = pad_sequence(list(map(LongTensor, true_words)))
        types = pad_sequence(list(map(LongTensor, types)))
        masked_indices = pad_sequence(list(map(LongTensor, masked_indices)))
        lens = list(map(len, sentences))
        word_pads = torch.ones(true_words.shape[0], true_words.shape[1], true_words.shape[1])
        for i, l in enumerate(lens):
            word_pads[i, :, l::] = 0
        return masked_words, true_words, types, word_pads, masked_indices

    return DataLoader(path, chunk_size, batch_size, post_processor)


def get_vocab_stats() -> two_ints:
    tokenizer = default_tokenizer()
    indexer = Indexer(tokenizer)
    return len(indexer.word_indices) + 1, len(indexer.type_indices)


def default_model() -> TypeFactoredLM:
    num_words, num_types = get_vocab_stats()
    d_model = 512
    d_ff = 1024
    d_k, d_v = d_model, d_model
    type_vocab_size, word_vocab_size = num_types, num_words
    num_layers = 8
    device = 'cuda'

    encoder_params = {'module_maker': EncoderLayer,
                      'num_layers': num_layers,
                      'num_heads': 8,
                      'd_model': d_model,
                      'd_ff': d_ff,
                      'd_k': d_k,
                      'd_v': d_v,
                      'activation_fn': F.gelu}
    type_pred_params = {'in_features': d_model, 'out_features': type_vocab_size}

    return TypeFactoredLM(masked_encoder=Encoder,
                          type_classifier=Linear,
                          num_words=word_vocab_size,
                          masked_encoder_kwargs=encoder_params,
                          type_classifier_kwargs=type_pred_params,
                          ).to(device)


def main():
    model = default_model()
    opt = torch.optim.AdamW(model.parameters())
    x_entropy = torch.nn.CrossEntropyLoss
    loss_kwargs = {'ignore_index': 0, 'reduction': 'mean'}
    loss_fn = MixedLoss(x_entropy, x_entropy, loss_kwargs, loss_kwargs, 0.15)

    dl = default_dataloader()

    num_epochs = 100
    num_sentences = 67010114
    batch_size = 128 
    nbatches_per_epoch = num_sentences // batch_size

    print('\nStarted training..') 
    sys.stdout.flush()
    for epoch in range(num_epochs):
        loss, s_acc, w_acc = train_batches(model, dl, loss_fn, opt, nbatches_per_epoch, 'cuda')
        print(loss, s_acc, w_acc)
        sys.stdout.flush()

if __name__ == "__main__": 
   main()