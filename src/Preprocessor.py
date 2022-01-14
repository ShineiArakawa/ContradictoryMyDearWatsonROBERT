import src.Parameters as Parameters
import string
import random
import numpy as np
import pandas as pd


class Preprocessor:
    """
    preprocessing unit
    """
    INDEX = "id"
    PREMISE = "premise"
    HYPOTHESIS = "hypothesis"
    LANG_AVB = "lang_avb"
    LANGUAGE = "language"
    LAVEL = "label"

    def __init__(self, trainDataPath='./data/train.csv', testDataPath='./data/test.csv', randomSeed=Parameters.RANDOM_SEED):
        self.trainDataPath = trainDataPath
        self.testDataPath = testDataPath
        self.csvTrainData = None
        self.csvTestData = None
        self.random = random
        if randomSeed is not None:
            self.random.seed(randomSeed)

    def prepareTrainAndEvalData(self, shuffle=False):
        data = []
        self.loadTrainData()
        for index, item in self.csvTrainData.iterrows():
            texts = []
            texts.append(self.cleanTexts(item[self.PREMISE]))
            texts.append(self.cleanTexts(item[self.HYPOTHESIS]))
            texts.append(item[self.LAVEL])
            data.append(texts)
        
        lenTrainData = int(len(data) * (1 - Parameters.RAITO_OF_EVAL_DATA))
        self.random.shuffle(data)
        trainData = data[:lenTrainData]
        evalData = data[lenTrainData:]

        return trainData, evalData

    def prepareTestData(self):
        data = []
        self.loadTestData()
        for index, item in self.csvTestData.iterrows():
            texts = []
            texts.append(item[self.INDEX])
            texts.append(self.cleanTexts(item[self.PREMISE]))
            texts.append(self.cleanTexts(item[self.HYPOTHESIS]))
            data.append(texts)
        return data

    def loadTrainData(self):
        with open(self.trainDataPath) as file:
            self.csvTrainData = pd.read_csv(file, header=0)
    
    def loadTestData(self):
        with open(self.testDataPath) as file:
            self.csvTestData = pd.read_csv(file, header=0)

    def cleanTexts(self, text):
        text = text.replace('\t', ' ')
        
        #for puctuation in string.punctuation:
        #    if (puctuation == '.') or (puctuation == ','):
        #        continue
        #    else:
        #        text = text.replace(puctuation, ' ')

        return text

