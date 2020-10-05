import torch
from torch import nn
from transformers import *

class BertEncoder(BertPreTrainedModel):
    def __init__(self, config, trainable=True):
        super(BertEncoder, self).__init__(config)
        self.config = config
        self.bert = BertModel(config)
        if not trainable:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False):
        seq_length = input_ids.size(-1)

        input_ids = input_ids.view(-1, seq_length)
        token_type_ids = token_type_ids.view(-1, seq_length)
        attention_mask = attention_mask.view(-1,seq_length)

        hidden, pooled_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        if not output_all_encoded_layers:
            last_hidden = pooled_output[0]
            pooled_output = pooled_output[0]
        return hidden, pooled_output

class UtteranceEncoder(nn.Module):
    def __init__(self, bert_dir='.', device=torch.device('cuda')):
        super(UtteranceEncoder, self).__init__()
        self.encoder = BertEncoder.from_pretrained(bert_dir, trainable=True)
        self.hidden_dim = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.device = device


    def forward(self, input_ids, token_ids, attention_mask):
        seq_length = input_ids.size(-1)
        hidden, pooled_output = self.encoder(input_ids.view(-1, seq_length),
                                             token_ids.view(-1, seq_length),
                                             attention_mask.view(-1,seq_length))
        hidden = self.dropout(hidden)
        pooled_output = self.dropout(hidden[:, 0, :])

        hidden = hidden.view(input_ids.size(0), input_ids.size(1), self.hidden_dim)
        pooled_output = pooled_output.view(input_ids.size(0), self.hidden_dim)

        return hidden, pooled_output