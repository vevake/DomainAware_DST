from models.utils import GELU
from torch import nn
import torch
class GetLogits(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_classes=1, dropout=0.1, none_emb=False, device=torch.device('cuda')):
        super(GetLogits, self).__init__()

        self.hidden_dim = hidden_dim
        self.linear_utterance = nn.Linear(input_dim, self.hidden_dim)
        self.l1 = nn.Linear(2*self.hidden_dim, self.hidden_dim)
        self.l2 = nn.Linear(self.hidden_dim, n_classes)
        self.activation = GELU()
        self.dropout = nn.Dropout(dropout)
        self.device = device

        if none_emb:
            self.none_emb = torch.rand(hidden_dim, requires_grad=True).to(device)
        else:
            self.none_emb = None

    def forward(self, inp, value_repn, mask=None, repeat=True):
        inp = self.activation(self.linear_utterance(inp))
        if self.none_emb is not None:
            none_repn = self.none_emb.expand(inp.size(0), 1, -1).to(self.device)
            value_repn = torch.cat([none_repn, value_repn[:, 1:, :]], dim=1)

        if repeat:
            repeat_inp = self.dropout(inp.unsqueeze(1).repeat(1, value_repn.size(1),1))
        else:
            repeat_inp = self.dropout(inp)

        value_repn = self.dropout(value_repn)
        score = self.l2(self.dropout(self.activation(self.l1(torch.cat([repeat_inp, value_repn], dim=2)))))

        if mask is not None:
            score[mask==0.]=-float('1e10')
        return score