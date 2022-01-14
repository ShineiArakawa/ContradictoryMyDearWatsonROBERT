from src.DataLoader import TestDataSet
import src.Parameters as Parameters
from src.Predicter import predict

import os
import datetime
import pandas as pd

import torch
from torch.utils.data import DataLoader


class Submitter:
    def __init__(self, dataSet):
        self.dataLoader = DataLoader(
            dataSet, batch_size=Parameters.TEST_BATCH_SIZE, shuffle=False)
        self.ids = dataSet.labels

    def makeFile(self, model, weightPath=None):
        if weightPath is not None:
            weights = torch.load(weightPath, map_location={'cuda:0': 'cpu'})
            model.load_state_dict(weights)

        outputLabels = predict(model=model, dataLoader=self.dataLoader)

        saveDirectoryPath = './submit_files'
        if not os.path.exists(saveDirectoryPath):
            os.makedirs(saveDirectoryPath)

        time_now = datetime.datetime.now()
        time_info = f'{time_now.year}-{time_now.month}-{time_now.day}_{time_now.hour}-{time_now.minute}-{time_now.second}'
        savePath = saveDirectoryPath + '/' + str(time_info) + '.csv'

        print(len(self.ids))
        print(len(outputLabels))
        dataFrame = pd.DataFrame(
            list(zip(self.ids, outputLabels)), columns=['id', 'prediction'])

        try:
            dataFrame.to_csv(savePath, index=False)
            print("Successed !!")
        except FileNotFoundError:
            print("Failed !!")
