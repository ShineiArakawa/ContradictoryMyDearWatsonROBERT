import transformers
from src.Preprocessor import Preprocessor
import src.Parameters as Parameters

import numpy as np
from numpy.core.fromnumeric import size
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as Func
from torch.utils.data import Dataset, DataLoader, dataset
from transformers import XLMRobertaTokenizer

class TrainDataSet(Dataset):
    def __init__(self, trainData, tokenizer):
        super(TrainDataSet, self).__init__()
        self.tokenizer = tokenizer
        self.trainData = trainData

    def __len__(self):
        return len(self.trainData)

    def __getitem__(self, index):
        return self.getData(index)

    def encodeToTokenIds(self, text):
        tokens = self.tokenizer.tokenize(text)
        return self.tokenizer.convert_tokens_to_ids(tokens=tokens)

    def getData(self, index):
        outputData = []
        premiseTokenIds = self.encodeToTokenIds(' '.join(self.trainData[index][0].split()))
        hypothesisTokenIds = self.encodeToTokenIds(' '.join(self.trainData[index][1].split()))
        labelIndex = torch.tensor(self.trainData[index][2], dtype=torch.long)
        
        inputs = self.tokenizer.encode_plus(     
            premiseTokenIds,
            hypothesisTokenIds,
            add_special_tokens=True,
            max_length=Parameters.MAX_LENGTH,
            padding = 'max_length',
            truncation=True,
            return_token_type_ids=True
        )

        outputData.append(torch.tensor(inputs['input_ids'], dtype=torch.long))
        outputData.append(torch.tensor(inputs['token_type_ids'], dtype=torch.long))
        outputData.append(torch.tensor(inputs['attention_mask'], dtype=torch.long))

        return outputData, labelIndex

    def getTokenizer(self):
        return self.tokenizer

class EvalDataSet(Dataset):
    def __init__(self, evalData, tokenizer):
        super(EvalDataSet, self).__init__()
        self.tokenizer = tokenizer
        self.evalData = evalData

    def __len__(self):
        return len(self.evalData)

    def __getitem__(self, index):
        return self.getData(index=index)

    def encodeToTokenIds(self, text):
        tokens = self.tokenizer.tokenize(text)
        return self.tokenizer.convert_tokens_to_ids(tokens=tokens)

    def getData(self, index):
        outputData = []
        premiseTokenIds = self.encodeToTokenIds(' '.join(self.evalData[index][0].split()))
        hypothesisTokenIds = self.encodeToTokenIds(' '.join(self.evalData[index][1].split()))
        labelIndex = torch.tensor(self.evalData[index][2], dtype=torch.long)
        
        inputs = self.tokenizer.encode_plus(     
            premiseTokenIds,
            hypothesisTokenIds,
            add_special_tokens=True,
            max_length=Parameters.MAX_LENGTH,
            padding = 'max_length',
            truncation=True,
            return_token_type_ids=True
        )

        outputData.append(torch.tensor(inputs['input_ids'], dtype=torch.long))
        outputData.append(torch.tensor(inputs['token_type_ids'], dtype=torch.long))
        outputData.append(torch.tensor(inputs['attention_mask'], dtype=torch.long))

        return outputData, labelIndex

class TestDataSet(Dataset):
    def __init__(self, testData, tokenizer):
        super(TestDataSet, self).__init__()
        self.tokenizer = tokenizer
        self.testData = testData
        self.labels = []

    def __len__(self):
        return len(self.testData)

    def __getitem__(self, index):
        return self.getData(index)

    def encodeToTokenIds(self, text):
        tokens = self.tokenizer.tokenize(text)
        return self.tokenizer.convert_tokens_to_ids(tokens=tokens)

    def getData(self, index):
        outputData = []
        self.labels.append(self.testData[index][0])
        premiseTokenIds = self.encodeToTokenIds(' '.join(self.testData[index][1].split()))
        hypothesisTokenIds = self.encodeToTokenIds(' '.join(self.testData[index][2].split()))
        
        inputs = self.tokenizer.encode_plus(     
            premiseTokenIds,
            hypothesisTokenIds,
            add_special_tokens=True,
            max_length=Parameters.MAX_LENGTH,
            padding = 'max_length',
            truncation=True,
            return_token_type_ids=True
        )

        outputData.append(torch.tensor(inputs['input_ids'], dtype=torch.long))
        outputData.append(torch.tensor(inputs['token_type_ids'], dtype=torch.long))
        outputData.append(torch.tensor(inputs['attention_mask'], dtype=torch.long))

        return outputData

class RoBertDataLoader:
    def __init__(self):
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(Parameters.MODEL_NAME)
        self.preprocessor = Preprocessor()
        
    def getTrainAndEvalDataLoader(self):
        trainData, evalData = self.preprocessor.prepareTrainAndEvalData(shuffle=True)

        trainDataSet = TrainDataSet(trainData=trainData, tokenizer=self.tokenizer)
        evalDataSet = EvalDataSet(evalData=evalData, tokenizer=self.tokenizer)
        
        trainDataLoader = DataLoader(trainDataSet, batch_size=Parameters.TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
        evalDataLoader = DataLoader(evalDataSet, batch_size=Parameters.TRAIN_BATCH_SIZE, shuffle=True)

        dataLoadersDict = {'train': trainDataLoader, 'eval': evalDataLoader}
        return dataLoadersDict

    def getTestDataLoader(self):
        testData = self.preprocessor.prepareTestData(shuffle=False)
        dataSet =  TestDataSet(testData=testData, tokenizer=self.tokenizer)
        testDataLoader = DataLoader(dataset, batch_size=Parameters.TEST_BATCH_SIZE, shuffle=False)
        return testDataLoader
