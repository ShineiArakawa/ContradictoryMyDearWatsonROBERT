import torch
import torch.nn as nn

class LossFunction(nn.Module):
    def __init__(self):
        super(LossFunction, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, inputTensor, lavelTensor):
        return self.loss(inputTensor, lavelTensor)