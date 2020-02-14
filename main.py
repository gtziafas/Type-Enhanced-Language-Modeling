from TypeLM.model.train import *
from TypeLM.model.eval import *
from TypeLM.model.masked_encoder import EncoderLayer, Encoder
from TypeLM.utils.utils import CustomLRScheduler, linear_scheme, save_model, load_model
from TypeLM.data.masker import default_masker, non_masker
from TypeLM.data.tokenizer import default_tokenizer, Indexer
import sys


def default_dataloader(path: str = '/data/s3913171/Lassy-Large/out.txt', chunk_size: int = 10240,
                       batch_size: int = 128, len_threshold: int = 100) -> LazyLoader:
    masker = default_masker()

    def post_processor(sentences: Samples) -> Tuple[LongTensor, LongTensor, LongTensor, LongTensor, LongTensor]:
        sentences = list(filter(lambda sentence: len(sentence[0]) < len_threshold, sentences))

        true_words, types = list(zip(*sentences))
        lens = list(map(len, true_words))
        masked_words, masked_indices = list(zip(*list(map(masker, true_words))))
        masked_words = pad_sequence(list(map(LongTensor, masked_words)))
        true_words = pad_sequence(list(map(LongTensor, true_words)))
        types = pad_sequence(list(map(LongTensor, types)))
        masked_indices = pad_sequence(list(map(LongTensor, masked_indices)))
        word_pads = torch.ones(true_words.shape[0], true_words.shape[1], true_words.shape[1])
        for i, l in enumerate(lens):
            word_pads[i, :, l::] = 0
        return masked_words, true_words, types, word_pads, masked_indices

    return LazyLoader(path, chunk_size, batch_size, post_processor)


def default_evaluator(path: str = '/data/s3913171/Lassy-Large/lassy_small.txt', batch_size: int = 128,
                      len_threshold: int = 100) -> EagerLoader:

    masker = non_masker()

    def post_processor(sentences: Samples) -> Tuple[LongTensor, LongTensor, LongTensor, LongTensor, LongTensor]:
        sentences = list(filter(lambda sentence: len(sentence[0]) < len_threshold, sentences))

        true_words, types = list(zip(*sentences))
        lens = list(map(len, true_words))
        masked_words, masked_indices = list(zip(*list(map(masker, true_words))))
        masked_words = pad_sequence(list(map(LongTensor, masked_words)))
        true_words = pad_sequence(list(map(LongTensor, true_words)))
        types = pad_sequence(list(map(LongTensor, types)))
        masked_indices = pad_sequence(list(map(LongTensor, masked_indices)))
        word_pads = torch.ones(true_words.shape[0], true_words.shape[1], true_words.shape[1])
        for i, l in enumerate(lens):
            word_pads[i, :, l::] = 0
        return masked_words, true_words, types, word_pads, masked_indices

    return EagerLoader(path, batch_size, post_processor)


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


def default_loss() -> MixedLoss:
    x_entropy = torch.nn.CrossEntropyLoss
    loss_kwargs = {'ignore_index': 0, 'reduction': 'mean'}
    return MixedLoss(x_entropy, x_entropy, loss_kwargs, loss_kwargs, 1)


def main(load_id: Optional[str], save_id: Optional[str]):

    model = default_model()

    batch_size = 128

    _opt = torch.optim.AdamW(model.parameters(), weight_decay=1e-03)
    opt = CustomLRScheduler(_opt, [linear_scheme], warmup_steps=100000, goal_lr=1e-04,
                            decrease_rate=1e-11, min_lr=1e-07)

    loss_fn = default_loss()
    # st_only_loss = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='mean')

    train_dl = default_dataloader(batch_size=batch_size)
    eval_dl = default_evaluator()

    num_epochs = 40
    num_sentences = 67010114
    num_batches_in_dataset = num_sentences // batch_size
    print_every = 1000
    num_minibatches_in_batch = num_batches_in_dataset // print_every

    if load_id is not None:
        model, opt, pre_train_epochs, _ = load_model(model_path=load_id, model=model, opt=opt)
        num_epochs += pre_train_epochs

    print('\nStarted training..') 
    sys.stdout.flush()
    for epoch in range(num_epochs * print_every):
        loss, s_acc, w_acc = train_batches(model, train_dl, loss_fn, opt, num_minibatches_in_batch, 'cuda')
        per = (epoch + 1) * num_minibatches_in_batch / num_batches_in_dataset
        print('\t' + ' '.join(['{:.2f}', '{:.4f}', '{:.4f}']).format(loss, s_acc, w_acc) + '\t' + '{:.3f}'.format(per))
        sys.stdout.flush()
        if not epoch % 10:
            print('-' * 64)
            print('\t {} steps'.format((epoch+1)*num_minibatches_in_batch))
            # remember: if using mixed loss, replace loss_fn by loss_fn.type_loss
            loss, s_acc, w_acc = eval_batches(model, eval_dl, loss_fn, 'cuda')
            print('\t' + ' '.join(['{:.2f}', '{:.4f}', '{:.4f}']).format(loss, s_acc, w_acc)
                  + '\t' + '{:.3f}'.format(per))
            print('-' * 64)
            sys.stdout.flush()
    if save_id is not None:
        save_model(model=model, save_id=save_id, opt=opt, num_epochs=num_epochs, loss=loss)

if __name__ == "__main__": 
    save_id = sys.argv[1] if len(sys.argv)>1 else None
    load_id = sys.argv[2] if len(sys.argv)>2 else None
    main(load_id, save_id)
