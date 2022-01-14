from src.DataLoader import TestDataSet
from src.Models import MainModel
from src.Submitter import Submitter
from src.Preprocessor import Preprocessor
import src.Parameters as Parameters 

import sys

from transformers import XLMRobertaTokenizer

def main():
    args = sys.argv
    weightPath = args[1]
    
    testData = Preprocessor().prepareTestData()
    tokenizer = XLMRobertaTokenizer.from_pretrained(Parameters.MODEL_NAME)
    dataSet = TestDataSet(testData=testData, tokenizer=tokenizer)
    model = MainModel()

    submitter = Submitter(dataSet=dataSet)
    submitter.makeFile(model=model, weightPath=weightPath)


if __name__ == '__main__':
    main()