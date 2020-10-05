import torch
from torch import nn
from models.utils import GELU
class CategoricalDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1, device=torch.device('cuda')):
        super(CategoricalDecoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.linear_utterance = nn.Linear(input_dim, self.hidden_dim)
        self.l1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.l2 = nn.Linear(self.hidden_dim, 1)
        self.activation = GELU()
        self.dropout = nn.Dropout(dropout)
        self.device = device

    def forward(self, inp, value_repn, mask=None):
        inp = self.dropout(self.activation(self.linear_utterance(inp)))

        value_repn = self.activation(self.l1(value_repn))
        value_repn = self.dropout(value_repn)

        score = torch.bmm(inp.view(-1, 1, self.hidden_dim), value_repn.view(-1, value_repn.size(-2), self.hidden_dim).transpose(1,2))

        score = score.view(inp.size(0), -1)
        return score