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

def main():
    saveParameters()
    model = MainModel()
    summary(model)
    dataLoader = RoBertDataLoader()
    dataLoadersDict = dataLoader.getTrainAndEvalDataLoader()
    criterion = LossFunction()

    modelParams = list(model.named_parameters())

    bertParams = [param for name, param in modelParams if 'bert' in name]
    classifierParams = [param for name, param in modelParams if 'bert' not in name]

    params = [
        {'params': bertParams, 'lr': Parameters.LEARNING_RATE},
        {'params': classifierParams, 'lr': Parameters.LEARNING_RATE * Parameters.LEARNING_RATE_COEFFICIENT}
    ]

    optimizer = AdamW(params, weight_decay=Parameters.WEIGHT_DECAY, correct_bias=False)
    trainedModel = trainModel(model=model, dataLoadersDict=dataLoadersDict, criterion=criterion, optimizer=optimizer)



if __name__ == '__main__':
    main()