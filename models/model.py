from models.encoder import UtteranceEncoder
from models.logitscorer import GetLogits
from models.MultiHeadAttention import MultiHeadAttention
from models.categoricalDecoder import CategoricalDecoder
from models.utils import GELU
from torch import nn
import torch

class Model(nn.Module):
    def __init__(self, bert_dir='.', device=torch.device('cuda')):
        super(Model, self).__init__()
        self.encoder = UtteranceEncoder(bert_dir, device)

        self.intent_classifier = GetLogits(self.encoder.hidden_dim, self.encoder.hidden_dim, n_classes=1, none_emb=True, device=device)
        self.req_slot_classifier = GetLogits(self.encoder.hidden_dim, self.encoder.hidden_dim, n_classes=1, none_emb=False, device=device)

        self.serv_attn = MultiHeadAttention(3, self.encoder.hidden_dim)
        self.slot_attn = MultiHeadAttention(3, self.encoder.hidden_dim)

        self.status_classifier = GetLogits(self.encoder.hidden_dim, self.encoder.hidden_dim, n_classes=3, device=device)
        self.cat_value_classifier = CategoricalDecoder(self.encoder.hidden_dim, self.encoder.hidden_dim, device=device)

        self.dropout = nn.Dropout(0.2)
        self.l_ctx = nn.Linear(self.encoder.hidden_dim*2, self.encoder.hidden_dim)
        self.l1 = nn.Linear(2*self.encoder.hidden_dim, self.encoder.hidden_dim)
        self.act = GELU()
        self.l2 = nn.Linear(self.encoder.hidden_dim, 2)

        self.device = device

    def forward(self, input_ids, token_ids, attention_mask, possible, masking):

        n_dials = input_ids.size(0)
        n_tokens = input_ids.size(1)

        n_cat_slots = possible['cat_slots'].size(-2)
        n_non_cat_slots = possible['non_cat_slots'].size(-2)
        n_values = possible['cat_values'].size(-2)
        n_req_slots = possible['req_slots'].size(-2)

        hidden, pooled_output = self.encoder(input_ids, token_ids, attention_mask)

        #INTENTS
        intent_score = self.intent_classifier(pooled_output, possible['intents'], masking['intents']).squeeze(-1)

        #service-specific repn
        hidden_serv = self.serv_attn(possible['services'], hidden, hidden, mask=attention_mask).squeeze(1)

        #REQ SLOT
        hidden_serv_req = self.dropout(hidden_serv.unsqueeze(1).repeat(1, n_req_slots, 1))
        hidden_req = hidden.unsqueeze(1).repeat(1, n_req_slots, 1, 1)
        attention_mask_tmp1 = attention_mask.unsqueeze(1).repeat(1, n_req_slots, 1).clone()
        hidden_req = hidden_req.view(-1, hidden_req.size(-2), self.encoder.hidden_dim)
        attention_mask_tmp1 = attention_mask_tmp1.view(-1, hidden_req.size(-2))

        slot_ctx_req = self.dropout(self.slot_attn(possible['req_slots'].view(-1, self.encoder.hidden_dim), hidden_req, hidden_req, mask=attention_mask_tmp1))
        slot_ctx_req = slot_ctx_req.view(n_dials, n_req_slots, self.encoder.hidden_dim)
        slot_ctx_req = self.dropout(self.l_ctx(torch.cat([slot_ctx_req, hidden_serv_req], dim=-1)))
        req_slot_score = self.req_slot_classifier(slot_ctx_req, possible['req_slots'], repeat=False).squeeze(-1)


        #CATEGORICAL SLOT
        hidden_serv_cat = self.dropout(hidden_serv.unsqueeze(1).repeat(1, n_cat_slots, 1))
        hidden_cat = hidden.unsqueeze(1).repeat(1, n_cat_slots, 1, 1)
        attention_mask_tmp = attention_mask.unsqueeze(1).repeat(1, n_cat_slots, 1).clone()
        hidden_cat = hidden_cat.view(-1, hidden_cat.size(-2), self.encoder.hidden_dim)
        attention_mask_tmp = attention_mask_tmp.view(-1, hidden_cat.size(-2))

        slot_ctx_cat = self.dropout(self.slot_attn(possible['cat_slots'].view(-1, self.encoder.hidden_dim), hidden_cat, hidden_cat, mask=attention_mask_tmp))
        slot_ctx_cat = slot_ctx_cat.view(n_dials, n_cat_slots, self.encoder.hidden_dim)
        slot_ctx_cat = self.dropout(self.l_ctx(torch.cat([slot_ctx_cat, hidden_serv_cat], dim=-1)))
        cat_status_score = self.status_classifier(slot_ctx_cat, possible['cat_slots'], repeat=False)

        cat_value_score = self.cat_value_classifier(slot_ctx_cat, possible['cat_values']).squeeze(-1)
        cat_value_score = cat_value_score.view(n_dials, n_cat_slots, n_values)
        cat_value_score[masking['cat_values']==0.] = -float('1e10')

        #NON-CATEGORICAL SLOT
        hidden_serv_noncat = self.dropout(hidden_serv.unsqueeze(1).repeat(1, n_non_cat_slots, 1))
        hidden_noncat = hidden.unsqueeze(1).repeat(1, n_non_cat_slots, 1, 1)
        attention_mask_tmp2 = attention_mask.unsqueeze(1).repeat(1, n_non_cat_slots, 1).clone()
        hidden_noncat = hidden_noncat.view(-1, hidden_noncat.size(-2), self.encoder.hidden_dim)
        attention_mask_tmp2 = attention_mask_tmp2.view(-1, hidden_noncat.size(-2))
        slot_ctx_noncat = self.dropout(self.slot_attn(possible['non_cat_slots'].view(-1, self.encoder.hidden_dim), hidden_noncat, hidden_noncat, mask=attention_mask_tmp2))
        slot_ctx_noncat = slot_ctx_noncat.view(n_dials, n_non_cat_slots, self.encoder.hidden_dim)
        slot_ctx_noncat = self.dropout(self.l_ctx(torch.cat([slot_ctx_noncat, hidden_serv_noncat], dim=-1))        )
        non_cat_status_score = self.status_classifier(slot_ctx_noncat, possible['non_cat_slots'], repeat=False)

        hidden_noncat = hidden.unsqueeze(1).repeat(1, n_non_cat_slots, 1, 1)
        possible['non_cat_slots'] = possible['non_cat_slots'].unsqueeze(2).repeat(1, 1, n_tokens, 1)
        non_cat_value_score = self.l2(self.dropout(self.act(self.l1(torch.cat([possible['non_cat_slots'], hidden_noncat], dim=-1)))))

        attention_mask_non_cat = attention_mask.unsqueeze(1).repeat(1, n_non_cat_slots, 1).clone()
        non_cat_value_score[attention_mask_non_cat==0.] = -float('1e10')

        return intent_score, req_slot_score, cat_status_score, cat_value_score, non_cat_status_score, non_cat_value_score