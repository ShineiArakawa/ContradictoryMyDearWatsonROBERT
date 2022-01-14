from src.Trainer import checkDevice

from tqdm import tqdm

import torch

def predict(model, dataLoader):
    predictions = []
    device = checkDevice()
    model.to(device)

    model.eval()

    for batch in tqdm(dataLoader):
        inputs = batch[0].to(device)
        tokenTypeIds = batch[1].to(device)

        outputs = model(input=inputs, tokenTypeIds=tokenTypeIds, attentionMask=None)
        _, prediction = torch.max(outputs, dim=1)
        prediction = prediction.flatten().tolist()
        predictions += prediction
    
    return predictions