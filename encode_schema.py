import torch
import numpy as np
from models.encoder import BertEncoder
from models.utils import get_bert_tokenizer
from data_reader.utils import _naive_tokenize
import pickle
import os, re
import config

def get_service_repn(model, tokenizer, service):
    PAD = 0
    n_service = len(service)
    item_ids = [[]]*n_service
    item_lens = [[]]*n_service
    max_len_tok = 0
    for serv, v in service.items():
        word_a = ["[CLS]"] + tokenizer.tokenize(' '.join(_naive_tokenize(serv))) + ["[SEP]"]
        word_b = tokenizer.tokenize(' '.join(_naive_tokenize(v['serv_desc']))) + ["[SEP]"]
        word_tok =  word_a + word_b
        word_tok = tokenizer.convert_tokens_to_ids(word_tok)
        max_len_tok = max(max_len_tok, len(word_tok))
        item_ids[v['serv_id']] = word_tok
        item_lens[v['serv_id']] = (len(word_a), len(word_tok))
    token_ids = np.zeros([n_service, max_len_tok])
    attn_mask = np.zeros([n_service, max_len_tok])
    for i, (a,b) in enumerate(item_lens):
        token_ids[i, a:b] = 1
        attn_mask[i, :b] = 1
    token_ids = torch.from_numpy(token_ids).long().to(config.DEVICE)
    attn_mask = torch.from_numpy(attn_mask).float().to(config.DEVICE)
    item_ids = [serv+[PAD]*(max_len_tok-len(serv)) for serv in item_ids]
    item_ids = torch.LongTensor(item_ids).to(config.DEVICE)
    REPN, _ = model(item_ids, token_ids, attn_mask, output_all_encoded_layers=True)
    REPN = REPN[:, 0, :].contiguous().view(item_ids.size(0), -1)
    REPN = REPN.detach().cpu().numpy()
    return REPN

def get_schema_repn(model, tokenizer, service, typ):
    PAD = 0
    n_service = len(service)
    item_ids = [[]]*n_service
    item_lens = [[]]*n_service
    max_len, max_len_tok = 0, 0
    for serv, v in service.items():
        names = v[typ]
        desc = v[typ+'_desc']
        serv_id = v['serv_id']
        seq = [[]] * len(names)
        seq_len = [[]] * len(names)
        for i, n in enumerate(names):
            if n == "NONE":
                word = 'None'
            else:
                word = ' '.join(re.split(r"([^a-zA-Z0-9])", n))
                word = ' '.join(word.split('_'))
            word_a = ["[CLS]"] + tokenizer.tokenize(word) + ["[SEP]"]
            word_b = tokenizer.tokenize(desc[n]) + ["[SEP]"]
            word_tok =  word_a + word_b
            word_tok = tokenizer.convert_tokens_to_ids(word_tok)
            max_len_tok = max(max_len_tok, len(word_tok))
            seq[names[n]] = word_tok
            seq_len[names[n]] = (len(word_a), len(word_tok))
        max_len = max(max_len, len(names))
        item_ids[serv_id] = seq
        item_lens[serv_id] = seq_len
    token_ids = np.zeros([n_service, max_len, max_len_tok])
    attn_mask = np.zeros([n_service, max_len, max_len_tok])
    MASK = np.zeros([n_service, max_len])
    for i, serv in enumerate(item_ids):
        MASK[i, :len(serv)] = 1.
        for j, (a,b) in enumerate(item_lens[i]):
            token_ids[i, j, a:b] = 1
            attn_mask[i, j, :b] = 1

    MASK = torch.from_numpy(MASK).float().to(config.DEVICE)
    token_ids = torch.from_numpy(token_ids).long().to(config.DEVICE)
    attn_mask = torch.from_numpy(attn_mask).float().to(config.DEVICE)
    item_ids = [[name+[PAD]*(max_len_tok-len(name)) for name in serv] for serv in item_ids]
    item_ids = [[name for name in serv]+[[PAD]*max_len_tok]*(max_len-len(serv)) for serv in item_ids]
    item_ids = torch.LongTensor(item_ids).to(config.DEVICE)
    REPN, _ = model(item_ids, token_ids, attn_mask, output_all_encoded_layers=True)
    REPN = REPN[:, 0, :].contiguous().view(item_ids.size(0), item_ids.size(1), -1)
    REPN = REPN * MASK.unsqueeze(-1)
    REPN = REPN.detach().cpu().numpy()
    MASK = MASK.detach().cpu().numpy()
    return REPN, MASK

def get_values_repn(model, tokenizer, services, cat_slots):
    PAD = tokenizer.convert_tokens_to_ids(['[PAD]'])[0]
    n_service = len(services)
    max_len_slots = max([len(v) for k, v in cat_slots.items()])
    max_len_values = max([len(v['values']) for s_name, serv in cat_slots.items() for s, v in serv.items()])
    max_len_tok = 0
    serv_seq = [[]] * n_service
    serv_seq_len = [[]] * n_service
    for serv, v in cat_slots.items():
        names = v
        desc = services[serv]['slots_desc']
        serv_id = services[serv]['serv_id']
        slot_seq = [[]] * max_len_slots
        slot_seq_len = [[]] * max_len_slots
        i = 0
        for i, n in enumerate(names):
            if n == "NONE":
                word = 'None'
            else:
                word = ' '.join(re.split(r"([^a-zA-Z0-9])", n))
                word = ' '.join(word.split('_'))

            val_seq = [[]] * max_len_values
            val_seq_len = [[]] * max_len_values
            for value, idx in v[n]['values'].items():

                value = ' '.join(value.split('_'))
                value = value.lower()
                word_a = ["[CLS]"] + tokenizer.tokenize(value) + ["[SEP]"]
                word_b = []
                word_tok =  word_a + word_b
                word_tok = tokenizer.convert_tokens_to_ids(word_tok)
                max_len_tok = max(max_len_tok, len(word_tok))
                val_seq[idx] = word_tok
                val_seq_len[idx] = (len(word_a), len(word_tok))

            slot_seq[names[n]['slot_id']] = val_seq
            slot_seq_len[names[n]['slot_id']] = val_seq_len

        if i < max_len_slots-1:
            val_seq = [[]] * max_len_values
            val_seq_len = [[(0,0)]] * max_len_values
            for j in range(len(names), max_len_slots):
                slot_seq[j] = val_seq
                slot_seq_len[j] = val_seq_len

        serv_seq[serv_id] = slot_seq
        serv_seq_len[serv_id] = slot_seq_len

    token_ids = np.zeros([n_service, max_len_slots, max_len_values, max_len_tok])
    attn_mask = np.zeros([n_service, max_len_slots, max_len_values, max_len_tok])
    MASK = np.zeros([n_service, max_len_slots, max_len_values])
    for i, serv in enumerate(serv_seq):
        for j, slot in enumerate(serv):
            for k, value in enumerate(slot):
                if len(value) > 0:
                    MASK[i, j, k] = 1.
                    a, b = serv_seq_len[i][j][k]
                    token_ids[i, j, k, a:b] = 1
                    attn_mask[i, j, k, :b] = 1.
    MASK = torch.from_numpy(MASK).float().to(config.DEVICE)
    token_ids = torch.from_numpy(token_ids).long().to(config.DEVICE)
    attn_mask = torch.from_numpy(attn_mask).float().to(config.DEVICE)
    item_ids = [[[value+[PAD]*(max_len_tok-len(value)) for value in slot]for slot in serv] for serv in serv_seq]
    item_ids = torch.LongTensor(item_ids).to(config.DEVICE)
    REPN, _ = model(item_ids, token_ids, attn_mask, output_all_encoded_layers=True)
    REPN = REPN[:, 0, :].contiguous().view([item_ids.size(0), item_ids.size(1), item_ids.size(2), -1])
    REPN = REPN * MASK.unsqueeze(-1)
    REPN = REPN.detach().cpu().numpy()
    MASK = MASK.detach().cpu().numpy()
    return REPN, MASK


if __name__ == "__main__":
    schemas_to_create = ['train', 'dev', 'test']

    model = BertEncoder.from_pretrained(os.path.join(config.BERT_DIR, config.BERT_MODEL))
    model.to(config.DEVICE)
    model.eval()

    tokenizer = get_bert_tokenizer()

    for s in schemas_to_create:
        with open(config.OUT_DIR + s + '_schema_dict.pkl', 'rb') as f:
            t = pickle.load(f)
        ALL_SERVICES_DICT = t['ALL_SERVICES_DICT']
        ALL_SERVICES_DICT_CAT = t['ALL_SERVICES_DICT_CAT']
        ALL_SERVICES_DICT_NONCAT = t['ALL_SERVICES_DICT_NONCAT']
        CAT_SLOTS_DICT = t['CAT_SLOTS_DICT']
        NON_CAT_SLOTS_DICT = t['NON_CAT_SLOTS_DICT']

        SERVICE_REPN = get_service_repn(model, tokenizer, ALL_SERVICES_DICT)
        INTENT_REPN, INTENT_MASK = get_schema_repn(model, tokenizer, ALL_SERVICES_DICT, typ='intents')
        SLOT_REPN, SLOT_MASK = get_schema_repn(model, tokenizer, ALL_SERVICES_DICT, typ='slots')

        CAT_SLOT_REPN, CAT_SLOT_MASK = get_schema_repn(model, tokenizer, ALL_SERVICES_DICT_CAT, typ='slots')
        NON_CAT_SLOT_REPN, NON_CAT_SLOT_MASK = get_schema_repn(model, tokenizer, ALL_SERVICES_DICT_NONCAT, typ='slots')

        CAT_VALUE_REPN, CAT_VALUE_MASK = get_values_repn(model, tokenizer, ALL_SERVICES_DICT, CAT_SLOTS_DICT)

        schema_embeddings = {'SERVICE_REPN':SERVICE_REPN,
            'INTENT_REPN': INTENT_REPN, 'INTENT_MASK': INTENT_MASK,
            'SLOT_REPN': SLOT_REPN, 'SLOT_MASK': SLOT_MASK,
            'CAT_SLOT_REPN': CAT_SLOT_REPN, 'CAT_SLOT_MASK': CAT_SLOT_MASK,
            'NON_CAT_SLOT_REPN': NON_CAT_SLOT_REPN, 'NON_CAT_SLOT_MASK': NON_CAT_SLOT_MASK,
            'CAT_VALUE_REPN': CAT_VALUE_REPN, 'CAT_VALUE_MASK': CAT_VALUE_MASK}
        torch.save(schema_embeddings, config.OUT_DIR + s + '_schema_emb.pt')