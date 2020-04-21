from nlp_nl.nl_eval.tasks import NER
from nlp_nl.prepare_data import get_ner
from TypeLM.finetuning.fineloaders import make_token_train_dl, make_token_test_dl
from TypeLM.finetuning.finetuners import TokenClassification, AdamW, CrossEntropyLoss
from tqdm import tqdm
from main import default_model
import sys

NUM_EPOCHS = 500


def main(data_folder: str, result_folder: str, model_path: str, device: str):
    get_ner(data_folder, result_folder)
    ner = NER(data_folder)
    trainset = list(map(lambda x: list(zip(*x[0])), tqdm(list(zip(ner.task.class_train_data)))))
    valset = list(map(lambda x: list(zip(*x[0])), tqdm(list(zip(ner.task.class_dev_data)))))
    testset = list(map(lambda x: list(zip(*x[0]))[0], tqdm(list(zip(ner.task.class_test_data)))))
    train_dl = make_token_train_dl(trainset, device=device)
    val_dl = make_token_train_dl(valset, device=device)
    test_dl = make_token_test_dl(testset, device=device)
    del trainset, valset, testset

    model = TokenClassification(default_model, model_path, len(ner.task.classes) + 1).to(device)
    optimizer = AdamW(model.parameters())
    loss_fn = CrossEntropyLoss(ignore_index=0, reduction='mean')

    for epoch in range(NUM_EPOCHS):
        print(('=' * 30) + f'{epoch}' + ('=' * 30))
        train_loss, train_acc = model.train_epoch(train_dl, optimizer, loss_fn)
        print(f'Train Loss: {train_loss}\t Train Acc: {train_acc}')
        val_loss, val_acc = model.eval_epoch(val_dl, loss_fn)
        print(f'Train Loss: {val_loss}\t Train Acc: {val_acc}')
        sys.stdout.flush()
