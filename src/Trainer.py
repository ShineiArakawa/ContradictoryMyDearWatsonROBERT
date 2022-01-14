from src.Models import MainModel
import src.Parameters as Parameters
from src.Utils import saveWeights, saveLogs

import time
from tqdm import tqdm
import numpy as np
from gradflowchecker import CheckGradFlow

import torch
import torchsummary
from transformers import get_linear_schedule_with_warmup


def trainModel(model: MainModel, dataLoadersDict, criterion, optimizer, toReturnLoss=False):
    device = checkDevice()
    model.to(device)
    criterion.to(device)
    torch.backends.cudnn.benchmark = True

    total_steps = len(dataLoadersDict['train'].dataset)*Parameters.MAX_EPOCH
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    gradFlowChecker = CheckGradFlow()

    trainLossLogs = np.zeros(shape=Parameters.MAX_EPOCH, dtype=np.float)
    evalLossLogs = np.zeros(shape=Parameters.MAX_EPOCH, dtype=np.float)
    trainAccuracyLogs = np.zeros(shape=Parameters.MAX_EPOCH, dtype=np.float)
    evalAccuracyLogs = np.zeros(shape=Parameters.MAX_EPOCH, dtype=np.float)

    for epoch in range(Parameters.MAX_EPOCH):
        for phase in ['train', 'eval']:
            print('-------------------------------------------------------------------------------------------------------------------------------------')
            print("Phase: ", phase)

            if phase == 'train':
                model.train()
            else:
                model.eval()

            epochLoss = 0.0
            epochCorrects = 0

            for batch in tqdm(dataLoadersDict[phase]):
                inputs = batch[0][0].to(device)
                tokenTypeIds = batch[0][1].to(device)
                attentionMask = batch[0][2].to(device)
                labels = batch[1].to(device)

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(
                        input=inputs, tokenTypeIds=tokenTypeIds, attentionMask=attentionMask)
                    loss = criterion(outputs, labels)
                    _, predictions = torch.max(outputs, dim=1)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        # gradFlowChecker.plotGrad(model.named_parameters())
                        # TODO I don"t know the effect of clipping grads
                        if Parameters.TO_USE_CLIP_GRAD:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

                    epochLoss += loss.item() * Parameters.TRAIN_BATCH_SIZE
                    epochCorrects += torch.sum(predictions == labels, dim=0)

            epochLoss = epochLoss / len(dataLoadersDict[phase].dataset)
            epochAccuracy = epochCorrects.double(
            ) / len(dataLoadersDict[phase].dataset)

            if phase == 'train':
                trainLossLogs[epoch] = epochLoss
                trainAccuracyLogs[epoch] = epochAccuracy
            else:
                evalLossLogs[epoch] = epochLoss
                evalAccuracyLogs[epoch] = epochAccuracy

            print('Epoch: {}/{}  |  Loss: {:.4f}  |  Acc: {:.4f}'.format(epoch +
                  1, Parameters.MAX_EPOCH, epochLoss, epochAccuracy))

    saveWeights(model=model)
    logs = [trainLossLogs, evalLossLogs, trainAccuracyLogs, evalAccuracyLogs]
    saveLogs(logs=logs)
    if toReturnLoss:
        return model, logs

    return model


def checkDevice():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Device: ", device)
    return device
