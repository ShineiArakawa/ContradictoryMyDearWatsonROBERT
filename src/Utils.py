import src.Parameters as Parameters

import os
import shutil
import datetime
import matplotlib.pyplot as plt

import torch

def saveWeights(model):
    saveDirectoryPath = './weight'
    if not os.path.exists(saveDirectoryPath):
        os.makedirs(saveDirectoryPath)

    time_now = datetime.datetime.now()
    time_info = f'{time_now.year}-{time_now.month}-{time_now.day}_{time_now.hour}-{time_now.minute}-{time_now.second}'
    savePath = saveDirectoryPath +'/'+ str(time_info) + '.pth'

    try:
        torch.save(model.state_dict(), savePath)
        print('Parameters were successfully saved!')
    except:
        print('Parameters were not successfully saved!')
    
    return None

def saveParameters():
    saveDirectoryPath = './Params'
    if not os.path.exists(saveDirectoryPath):
        os.makedirs(saveDirectoryPath)

    time_now = datetime.datetime.now()
    time_info = f'{time_now.year}-{time_now.month}-{time_now.day}_{time_now.hour}-{time_now.minute}-{time_now.second}'
    savePath = saveDirectoryPath +'/'+ str(time_info) + '.txt'

    shutil.copy2('./src/Parameters.py', savePath)

def saveLogs(logs):
    time_now = datetime.datetime.now()
    time_info = f'{time_now.year}-{time_now.month}-{time_now.day}_{time_now.hour}-{time_now.minute}-{time_now.second}'

    saveDirectoryPath = './logs/' + str(time_info)
    if not os.path.exists(saveDirectoryPath):
        os.makedirs(saveDirectoryPath)

    savePathLoss = saveDirectoryPath  + '/loss' + '.jpg'
    savePathAcurracy = saveDirectoryPath  + '/accuracy' + '.jpg'

    x = [num for num in range(Parameters.MAX_EPOCH)]
    epochTrainLosses = logs[0].tolist()
    epochEvalLosses = logs[1].tolist()
    epochTrainAccuracies = logs[2].tolist()
    epochEvalAccuracies = logs[3].tolist()

    # Loss
    plt.plot(x, epochTrainLosses, color='red', label='Train Loss')
    plt.plot(x, epochEvalLosses, color='blue', label='Eval Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid()
    

    plt.savefig(savePathLoss)
    plt.clf()

    # Accuracy
    plt.plot(x, epochTrainAccuracies, color='red', label='Train Accuracy')
    plt.plot(x, epochEvalAccuracies, color='blue', label='Eval Accuracy')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid()
 
    plt.savefig(savePathAcurracy)
    
    return None