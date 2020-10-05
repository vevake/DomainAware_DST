import torch
from torch import nn
import os
import math
import config
from transformers import *
class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

def get_bert_tokenizer():
    tokenizer = BertTokenizer.from_pretrained(os.path.join(config.BERT_DIR, config.BERT_MODEL, 'vocab.txt'), do_lower_case=False)
    return tokenizer