import torch.nn as nn


def loss_function(inp, out):
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.MSELoss()
    return criterion(out, inp)
