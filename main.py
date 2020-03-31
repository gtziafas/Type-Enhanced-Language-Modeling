from TypeLM.model.train import *
from TypeLM.model.eval import *
from TypeLM.model.tensor_fusion import *
from TypeLM.model.masked_encoder import EncoderLayer, Encoder
from TypeLM.utils.utils import CustomLRScheduler, linear_scheme, save_model, load_model, ElementWiseFusion
from TypeLM.model.loss import FuzzyLoss, CrossEntropySS
from TypeLM.data.masker import default_word_masker, non_masker, default_type_masker
from TypeLM.data.tokenizer import default_tokenizer, Indexer
import sys
import argparse


def default_dataloader(path: str = '/data/s3913171/Lassy-Large/out.txt', chunk_size: int = 10240,
                       batch_size: int = 128, len_threshold: int = 100) -> LazyLoader:
    word_masker = default_word_masker()
    type_masker = default_type_masker()

    def post_processor(sentences: Samples) \
            -> Tuple[LongTensor, LongTensor, LongTensor, LongTensor, LongTensor, LongTensor]:
        sentences = list(filter(lambda sentence: len(sentence[0]) < len_threshold, sentences))

        true_words, true_types = list(zip(*sentences))
        lens = list(map(len, true_words))
        masked_words, masked_indices = list(zip(*list(map(word_masker, true_words))))
        masked_types, _ = list(zip(*list(map(type_masker, true_types))))
        masked_words = pad_sequence(list(map(LongTensor, masked_words)))
        true_words = pad_sequence(list(map(LongTensor, true_words)))
        true_types = pad_sequence(list(map(LongTensor, true_types)))
        masked_types = pad_sequence(list(map(LongTensor, masked_types)))
        masked_indices = pad_sequence(list(map(LongTensor, masked_indices)))
        word_pads = torch.ones(true_words.shape[0], true_words.shape[1], true_words.shape[1])
        for i, l in enumerate(lens):
            word_pads[i, :, l::] = 0
        return masked_words, true_words, true_types, masked_types, word_pads, masked_indices

    return LazyLoader(path, chunk_size, batch_size, post_processor)


def default_evaluator(path: str = '/data/s3913171/Lassy-Large/lassy_small.txt', batch_size: int = 128,
                      len_threshold: int = 100) -> EagerLoader:

    masker = non_masker()

    def post_processor(sentences: Samples) -> Tuple[LongTensor, LongTensor, LongTensor, LongTensor, LongTensor]:
        sentences = list(filter(lambda sentence: len(sentence[0]) < len_threshold, sentences))

        true_words, true_types = list(zip(*sentences))
        lens = list(map(len, true_words))
        masked_words, masked_indices = list(zip(*list(map(masker, true_words))))
        masked_words = pad_sequence(list(map(LongTensor, masked_words)))
        true_words = pad_sequence(list(map(LongTensor, true_words)))
        true_types = pad_sequence(list(map(LongTensor, true_types)))
        masked_indices = pad_sequence(list(map(LongTensor, masked_indices)))
        word_pads = torch.ones(true_words.shape[0], true_words.shape[1], true_words.shape[1])
        for i, l in enumerate(lens):
            word_pads[i, :, l::] = 0
        return masked_words, true_words, true_types, word_pads, masked_indices

    return EagerLoader(path, batch_size, post_processor)


def get_vocab_stats() -> two_ints:
    tokenizer = default_tokenizer()
    indexer = Indexer(tokenizer)
    return len(indexer.word_indices) + 1, len(indexer.type_indices)


def conv2dfusion_model() -> TypeFactoredLM:
    num_words, num_types = get_vocab_stats()
    d_model = 512
    d_ff = 1024
    d_k, d_v = d_model, d_model
    num_layers = 8
    num_heads = 8
    device = 'cuda'

    encoder_params = {'module_maker': EncoderLayer,
                      'num_layers': num_layers,
                      'num_heads': num_heads,
                      'd_model': d_model,
                      'd_ff': d_ff,
                      'd_k': d_k,
                      'd_v': d_v,
                      'activation_fn': F.gelu}
    type_pred_params = {'in_features': d_model, 'out_features': num_types}
    label_smoother_params = {'smoothing': 0.1, 'num_classes': num_types}

    fusion = Conv2dFusion 
    deep_params = get_deep_params()
    # shallow_params = get_shallow_params()
    fusion_params = {'fusion': Outter2dFusion, 'conv': Conv2dFeatures, 'fusion_kwargs':{}, 'conv_kwargs':deep_params}
    # fusion_params = {'fusion': Outter2dFusion, 'conv': Conv2dFeatures, 'fusion_kwargs':{}, 'conv_kwargs':shallow_params}

    return TypeFactoredLM(masked_encoder=Encoder,
                          type_classifier=Linear,
                          num_words=num_words,
                          masked_encoder_kwargs=encoder_params,
                          type_classifier_kwargs=type_pred_params,
                          fusion=fusion,
                          fusion_kwargs=fusion_params,
                          type_embedder=Linear,
                          type_embedder_kwargs={'in_features': num_types, 'out_features': d_model},
                          label_smoother_kwargs=label_smoother_params
                          ).to(device)


def default_model() -> TypeFactoredLM:
    num_words, num_types = get_vocab_stats()
    d_model = 512
    d_ff = 1024
    d_k, d_v = d_model, d_model
    num_layers = 8
    num_heads = 8
    device = 'cuda'

    encoder_params = {'module_maker': EncoderLayer,
                      'num_layers': num_layers,
                      'num_heads': num_heads,
                      'd_model': d_model,
                      'd_ff': d_ff,
                      'd_k': d_k,
                      'd_v': d_v,
                      'activation_fn': F.gelu}
    type_pred_params = {'in_features': d_model, 'out_features': num_types}
    label_smoother_params = {'smoothing': 0.1, 'num_classes': num_types}

    return TypeFactoredLM(masked_encoder=Encoder,
                          type_classifier=Linear,
                          num_words=num_words,
                          masked_encoder_kwargs=encoder_params,
                          type_classifier_kwargs=type_pred_params,
                          fusion=ElementWiseFusion,
                          fusion_kwargs={'activation': F.tanh},
                          type_embedder=Linear,
                          type_embedder_kwargs={'in_features': num_types, 'out_features': d_model},
                          label_smoother_kwargs=label_smoother_params
                          ).to(device)


def default_loss() -> MixedLoss:
    mlm_loss_kwargs = {'ignore_index': 0, 'reduction': 'mean'}
    st_loss_kwargs = {'num_classes': get_vocab_stats()[1], 'mass_redistribution': 0.1, 'ignore_index': 0,
                      'reduction': 'batchmean'}
    return MixedLoss(CrossEntropySS, FuzzyLoss, mlm_loss_kwargs, st_loss_kwargs, 2)


def main(load_id: Optional[str], save_id: Optional[str]):

    model = default_model()

    batch_size = 128
    
    loss_fn = default_loss()

    train_dl = default_dataloader(batch_size=batch_size)
    eval_dl = default_evaluator()

    num_epochs = 1
    num_sentences = 67010114
    num_batches_in_dataset = num_sentences // batch_size
    steps_per_epoch = 1000
    num_minibatches_in_batch = num_batches_in_dataset // steps_per_epoch

    _opt = torch.optim.AdamW(model.parameters(), weight_decay=1e-07)
    pre_train_epochs = 0 
    if load_id is not None:
        model, _opt, pre_train_epochs, _ = load_model(model_path=load_id, model=model, opt=_opt)
    opt = CustomLRScheduler(_opt, [linear_scheme], warmup_steps=1e05, goal_lr=5e-05, decrease_rate=1e-11, min_lr=1e-07)
    opt._step = pre_train_epochs * num_batches_in_dataset
    
    print('\nStarted training..') 
    sys.stdout.flush()
    for step in range(num_epochs * steps_per_epoch):
        step_stats = train_batches(model, train_dl, loss_fn, opt, num_minibatches_in_batch, 'cuda')
        (mlm_loss, st_loss), s_acc, w_acc = step_stats
        per = (step + 1) * num_minibatches_in_batch / num_batches_in_dataset
        print(f' MLM: {mlm_loss:.4f}\tST: {st_loss:.4f}\tS_acc: {(s_acc*100):.2f}'
              f'\tW_acc: {(w_acc*100):.2f}\tStep: {(per*100):.2f}'
              f'\topt_steps: {opt._step:d}\topt_lr:{opt.lrs[-1]:.10f}')
        sys.stdout.flush()
        if not step % 50:
            print('-' * 64)
            # remember: if using mixed loss, replace loss_fn by loss_fn.type_loss
            st_loss, s_acc, w_acc = eval_batches(model, eval_dl, loss_fn.type_loss, 'cuda')
            print(f'\tST: {st_loss:.4f}\tS_acc: {(s_acc * 100):.2f}\tW_acc: {(w_acc * 100):.2f}')
            print(f'Supertagging Layer: {model.layer_weighter.layer_weights.softmax(dim=0).tolist()}')
            print('-' * 64)
            sys.stdout.flush()
    if save_id is not None:
        save_model(model=model, save_id=save_id, opt=_opt, num_epochs=pre_train_epochs + num_epochs,
                   loss=(mlm_loss, st_loss))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--save_id', help='where to save the model once training ends', type=str)
    parser.add_argument('-l', '--load_id', help='which model to start loading from', type=str)

    kwargs = vars(parser.parse_args())
    main(**kwargs)
