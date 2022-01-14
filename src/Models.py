import torch
import src.Parameters as Parameters

import torch.nn as nn
from transformers import XLMRobertaConfig, XLMRobertaModel

class MainModel(nn.ModuleList):
    def __init__(self):
        super(MainModel, self).__init__()
        self.bertModel = SingleROBERT(toUseTailTokenOutput=Parameters.TO_USE_TAIL_TOKEN_OUTPUT)
        bertConfig = self.bertModel.getConfig()
        hiddenSize = bertConfig.hidden_size
        
        self.classifier = SimpleClassifier(hiddenSize=hiddenSize, nClasses=Parameters.NUM_OF_CLASSES)

    def forward(self, input, tokenTypeIds, attentionMask):
        output = self.bertModel(input=input, tokenTypeIds=tokenTypeIds, attentionMask=attentionMask)
        output = self.classifier(output)
        return output 

class SingleROBERT(nn.Module):
    def __init__(self, toUseTailTokenOutput=False):
        super(SingleROBERT, self).__init__()
        self.bertConfig = XLMRobertaConfig.from_pretrained(Parameters.MODEL_NAME)
        self.bertModel = XLMRobertaModel.from_pretrained(Parameters.MODEL_NAME)
        self.toUseTailTokenOutput = toUseTailTokenOutput
        self.pooling = nn.AdaptiveMaxPool1d(output_size=self.bertConfig.hidden_size)
        
        for name, param in self.bertModel.named_parameters():
            if 'pool' in name:
                param.requires_grad = False    
            else:
                param.requires_grad = True

    def forward(self, input, tokenTypeIds, attentionMask):
        _, _, hiddenStates = self.bertModel(input_ids=input, attention_mask=attentionMask, token_type_ids=tokenTypeIds, return_dict=False, output_hidden_states=True)
        outputHead = hiddenStates[-1][:, 0, :]
        if Parameters.NUM_HIDDEN_STATES_TO_USE > 1:
            for i in range(1, Parameters.NUM_HIDDEN_STATES_TO_USE):
                outputHead = torch.cat((outputHead, hiddenStates[-i-1][:, 0, :]), dim=1)
            
            if self.toUseTailTokenOutput:
                tailTokenOutPut = hiddenStates[-1][:, Parameters.MAX_LENGTH-1, :]
                for i in range(1, Parameters.NUM_HIDDEN_STATES_TO_USE):
                    tailTokenOutPut = torch.cat((tailTokenOutPut, hiddenStates[-i-1][:, Parameters.MAX_LENGTH-1, :]), dim=1)

                output = torch.cat((outputHead, tailTokenOutPut), dim=1)
                output = output.unsqueeze(dim=1)
                output = self.pooling(output)
                output = output.squeeze()
            else:
                outputHead = outputHead.unsqueeze(dim=1)
                output = self.pooling(outputHead)
                output = output.squeeze()
        else:
            if self.toUseTailTokenOutput:
                tailTokenOutPut = hiddenStates[-1][:, Parameters.MAX_LENGTH-1, :]
                output = torch.cat((outputHead, tailTokenOutPut), dim=1)
                output = output.unsqueeze(dim=1)
                output = self.pooling(output)
                output = output.squeeze()
            else:
                output = outputHead

        return output

    def getConfig(self):
        return self.bertConfig

class Classifier(nn.Module):
    def __init__(self, hiddenSize, nClasses, dropoutRate):
        super(Classifier, self).__init__()
        self.dropout1 = nn.Dropout(p=dropoutRate)
        self.linear1 = nn.Linear(in_features=hiddenSize, out_features=hiddenSize)
        self.batchNorm = nn.BatchNorm1d(num_features=hiddenSize, eps=1e-05, momentum=0.1, affine=False)
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.dropout2 = nn.Dropout(p=dropoutRate)
        self.linear2 = nn.Linear(in_features=hiddenSize, out_features=nClasses)

        nn.init.normal_(self.linear1.weight, std=0.04)
        #nn.init.xavier_normal_(self.linear2.weight)
        nn.init.normal_(self.linear2.weight, mean=0.5, std=0.04)
        nn.init.normal_(self.linear1.bias, 0)
        nn.init.normal_(self.linear2.bias, 0)

    def forward(self, input):
        output = self.dropout1(input)
        output = self.linear1(output)
        output = self.batchNorm(output)
        output = self.activation(output)
        output = self.dropout2(output)
        output = self.linear2(output)
        return output

class SimpleClassifier(nn.Module):
    def __init__(self, hiddenSize, nClasses):
        super(SimpleClassifier, self).__init__()
        self.linear1 = nn.Linear(in_features=hiddenSize, out_features=nClasses)

        nn.init.normal_(self.linear1.weight, std=0.03)
        nn.init.constant_(self.linear1.bias, 0)

    def forward(self, input):
        output = self.linear1(input)
        return output