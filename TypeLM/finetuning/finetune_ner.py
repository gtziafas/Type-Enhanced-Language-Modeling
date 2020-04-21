from nlp_nl.nl_eval.tasks import NER
from nlp_nl.prepare_data import get_ner
from TypeLM.finetuning.fineloaders import make_token_train_dl, make_token_test_dl
from TypeLM.finetuning.finetuners import TokenClassification, AdamW, CrossEntropyLoss
from TypeLM.utils.imports import *
from TypeLM.utils.utils import save_model
from tqdm import tqdm
from main import default_model
import sys
import pickle

NUM_EPOCHS = 500


def prepare_data(data_folder: str, result_folder: str, device: str, save_to: str):
    get_ner(data_folder, result_folder)
    ner = NER(data_folder)
    trainset = list(map(lambda x: list(zip(*x[0])), tqdm(list(zip(ner.task.class_train_data)))))
    valset = list(map(lambda x: list(zip(*x[0])), tqdm(list(zip(ner.task.class_dev_data)))))
    testset = list(map(lambda x: list(zip(*x[0]))[0], tqdm(list(zip(ner.task.class_test_data)))))
    train_dl = make_token_train_dl(trainset, device=device)
    print('Done making train loader.')
    sys.stdout.flush()
    val_dl = make_token_train_dl(valset, device=device)
    print('Done making val loader.')
    sys.stdout.flush()
    test_dl = make_token_test_dl(testset, device=device)
    print('Done making test loader.')
    sys.stdout.flush()
    del trainset, valset, testset
    with open(save_to, 'wb') as f:
        pickle.dump([ner, train_dl, val_dl, test_dl], f)


def main(load_from: str, model_path: str, device: str, save_id: str):
    with open(load_from, 'rb') as f:
        ner, train_dl, val_dl, test_dl = pickle.load(f)

    model = TokenClassification(default_model, model_path, len(ner.task.classes) + 1).to(device)
    optimizer = AdamW(model.parameters())
    loss_fn = CrossEntropyLoss(ignore_index=0, reduction='mean')

    best_acc = -1

    for epoch in range(NUM_EPOCHS):
        print(('=' * 30) + f'{epoch}' + ('=' * 30))
        train_loss, train_acc = model.train_epoch(train_dl, optimizer, loss_fn)
        print(f'Train Loss: {train_loss}\t Train Acc: {train_acc}')
        val_loss, val_acc = model.eval_epoch(val_dl, loss_fn)
        print(f'Val Loss: {val_loss}\t Val Acc: {val_acc}')
        sys.stdout.flush()
        print('Dont look below')
        predictions = model.infer_dataloader(test_dl)
        print(ner.task.predict_items(lambda x: predictions))
        sys.stdout.flush()
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({'model_state_dict': model.state_dict()}, save_id)

    model.load_state_dict(torch.load(save_id)['model_state_dict'])
    predictions = model.infer_dataloader(test_dl)
    print(ner.task.predict_items(lambda x: predictions))
    sys.stdout.flush()
