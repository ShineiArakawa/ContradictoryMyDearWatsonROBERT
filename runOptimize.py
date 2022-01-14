from torch.nn.parameter import Parameter
from src.DataLoader import RoBertDataLoader
from src.Models import MainModel
from src.LossFunction import LossFunction
from src.Utils import saveParameters
from src.Trainer import trainModel
import src.Parameters as Parameters

import numpy as np

from torch.optim import Adam
from transformers import AdamW
from torchsummary import summary

import optuna


def main():
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)
    print("best_value= ", study.best_value)
    print("best_params= ", study.best_params)
    print("="*80)
    for trial in study.get_trials():
        print(
            f"{trial.number}: {trial.value:.3f}, backboneLR= {trial.params['backboneLR']}, classifierLR= {trial.params['classifierLR']}")
    pass


def objective(trial: optuna.Trial):
    lr1 = trial.suggest_loguniform("backboneLR", 1e-4, 0.1)
    lr2 = trial.suggest_loguniform("classifierLR", 1e-3, 0.1)
    
    saveParameters()
    model = MainModel()
    summary(model)
    dataLoader = RoBertDataLoader()
    dataLoadersDict = dataLoader.getTrainAndEvalDataLoader()
    criterion = LossFunction()

    modelParams = list(model.named_parameters())

    bertParams = [param for name, param in modelParams if 'bert' in name]
    classifierParams = [param for name,
                        param in modelParams if 'bert' not in name]

    params = [
        {'params': bertParams, 'lr': lr1},
        {'params': classifierParams, 'lr': lr2}
    ]

    optimizer = AdamW(
        params, weight_decay=Parameters.WEIGHT_DECAY, correct_bias=False)
    trainedModel, logs = trainModel(
        model=model, dataLoadersDict=dataLoadersDict, criterion=criterion, optimizer=optimizer, toReturnLoss=True)

    evalLoss = logs[1][-1]

    return evalLoss


if __name__ == '__main__':
    main()
