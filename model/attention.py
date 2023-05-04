import torch
import torch.nn as nn

class AttentionLayer(nn.Module):
    def __init__(self, device, hidden_size=64, bidirectional=True):
        super().__init__()
        self.device = device
        self.hidden_size = (hidden_size*2 if bidirectional else hidden_size)
        self.linear_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.u_context = nn.Parameter(torch.FloatTensor(self.hidden_size).normal_(0, 0.01)) # shape(hidden_size)

    def forward(self, x, x_lengths):
        if isinstance(x_lengths, list):
            x_lengths = torch.LongTensor(x_lengths).to(self.device)
        h = self.tanh(self.linear_layer(x)) # shape(bs, longest_seq_len, hidden_size)
        alpha = torch.mul(h, self.u_context).sum(dim=2) # shape(bs, longest_seq_len)
        max_len = alpha.size(1)
        mask = torch.arange(max_len)[None,:].to(self.device) < x_lengths[:,None]
        alpha[~mask] = float('-inf')
        alpha = self.softmax(alpha) # shape(bs, longest_seq_len)
        attention_output = torch.bmm(x.transpose(1, 2), alpha.unsqueeze(2)).squeeze(2)
        return attention_output, alpha

