import torch.nn as nn

def init_linear_layer(layer):
    if type(layer) == nn.Linear:
        nn.init.xavier_uniform(layer.weight)
        layer.bias.data.fill_(0.01)

def init_rnn(rnn):
    for name, param in rnn.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0.0)
        elif 'weight' in name:
            nn.init.xavier_uniform_(param)