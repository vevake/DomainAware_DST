import os
import copy
import torch
import random
import json
import numpy as np
from data_reader.utils import _naive_tokenize, _tokenize, _find_subword_indices
import config
import warnings
warnings.filterwarnings("ignore")

def get_files(typ='train'):
    if typ == 'train':
        files = [os.path.join(config.DATA_DIR, 'train', f) for f in os.listdir(config.DATA_DIR+'train/') if f.split('_')[0]=='dialogues']
    elif typ == 'dev':
        files = [os.path.join(config.DATA_DIR, 'dev', f) for f in os.listdir(config.DATA_DIR+'dev/') if f.split('_')[0]=='dialogues']
    elif typ =='test':
        files = [os.path.join(config.DATA_DIR, 'test', f) for f in os.listdir(config.DATA_DIR+'test/') if f.split('_')[0]=='dialogues']
    else:
        print('file type error')
        exit()
    return files


def get_dialogues():
    train_files = get_files('train')
    dev_files = get_files('dev')
    test_files = get_files('test')

    if config.DOMAIN.lower() =='single':
        train_files = [d for d in train_files if int(d.split('/')[-1].split('_')[1].strip('.json')) < 44]
    elif config.DOMAIN.lower() == 'multi':
        train_files = [d for d in train_files if int(d.split('/')[-1].split('_')[1].strip('.json')) >= 44]
    elif config.DOMAIN.lower() == 'all':
        pass
    else:
        print('Error in domain type specified.')
        exit()

    train_dialogues = get_data(train_files)
    dev_dialogues = get_data(dev_files)
    test_dialogues = get_data(test_files)

    return train_dialogues, dev_dialogues, test_dialogues

class InputExample(object):
    def __init__(self, t_id, t_utterance, frames, prev):
        self.t_id = t_id
        self.utterance = t_utterance
        self.frames = frames
        self.prev = prev

def get_data(dialogue_files):
    all_dials = []
    for f in dialogue_files:
        dialogues = json.load(open(f, 'r'))
        for dial in dialogues:
            d = []
            prev = {'sys_res': '', 'frames': {}}
            d_id = dial['dialogue_id']
            d_services = dial['services']

            prev_state = {serv: {} for serv in d_services}
            for t_id, turn in enumerate(dial['turns']):
                t_speaker = turn['speaker']
                t_utterance = turn['utterance']
                curr_turn_frames = []
                if t_speaker == 'SYSTEM':
                    prev['sys_res'] = t_utterance
                for frame in turn['frames']:
                    if t_speaker == 'SYSTEM':
                        prev['frames'][frame['service']] = frame

                    elif t_speaker == 'USER':
                        frames = {'service': '', 'active_intent': '', 'slots': [], 'turn_states':[], 'states': [], 'req_slots': []}
                        f_service = frame['service']

                        frames['service'] = f_service
                        frames['slots'] = frame['slots']
                        st = frame['state']
                        turn_slot_pred, state = [], []
                        for s, v in st['slot_values'].items():
                            if s not in prev_state[f_service]:
                                turn_slot_pred.append((s, v))
                            else:
                                if prev_state[f_service][s][0] not in v:
                                    turn_slot_pred.append((s, v))
                            state.append((s, v))
                        prev_state[f_service] = st['slot_values']
                        frames['active_intent'] = st['active_intent']
                        frames['states'] = state
                        frames['turn_states'] = turn_slot_pred
                        frames['req_slots'] = st['requested_slots']
                        curr_turn_frames.append(copy.deepcopy(frames))
                if t_speaker == 'USER':
                    d.append(InputExample(t_id, t_utterance, curr_turn_frames, copy.deepcopy(prev)))
            all_dials.append({'dialogue_id': d_id, 'services': d_services, 'turns': d})

    return all_dials

def get_seqs(dialogues, tokenizer, schema_dict, schema_emb, typ ='train'):
    dc, ot =0, 0
    n_cat_slots = schema_emb['CAT_SLOT_REPN'][typ].shape[-2]
    n_non_cat_slots = schema_emb['NON_CAT_SLOT_REPN'][typ].shape[-2]
    id_dial, utt_true_dial, x_dial, x_dial_len, y_dial, inv_align_dial, prev_sys_frame_dial = [], [], [], [], [], [], []
    id_turn, utt_true_turn, x_turn, x_turn_len, y_turn, inv_align_turn, prev_sys_frame_turn = [], [], [], [], [], [], []
    for dial in dialogues:

        d_id = dial['dialogue_id']
        dial_services = dial['services']

        for turn in dial['turns']:
            sys_res_tok, sys_res_align, sys_res_inv_align = _tokenize(tokenizer, turn.prev['sys_res'])
            prev_res = ["[CLS]"] + sys_res_tok + ["[SEP]"]

            utterance_tok, utterance_align, utterance_inv_align = _tokenize(tokenizer, turn.utterance)
            utterance = utterance_tok + ["[SEP]"]

            utt_true_turn.append([turn.prev['sys_res'], turn.utterance])
            l = [len(prev_res), len(utterance)]

            x = tokenizer.convert_tokens_to_ids(prev_res + utterance)
            x_turn.append(x)
            x_turn_len.append(l)

            frames = {}
            for f in turn.frames:
                serv = schema_dict['ALL_SERVICES_DICT'][typ][f['service']]['serv_id']
                frames[serv] = {'active_intent': 0, 'req_slots': [], 'cat_slots': [-1]*n_cat_slots, 'non_cat_slots': {'status':[-1]*n_non_cat_slots, 'start':[-1]*n_non_cat_slots, 'end':[-1]*n_non_cat_slots}}

                frames[serv]['active_intent'] = schema_dict['ALL_SERVICES_DICT'][typ][f['service']]['intents'][f['active_intent']]
                frames[serv]['req_slots'] = [schema_dict['ALL_SERVICES_DICT'][typ][f['service']]['slots'][a] for a in f['req_slots']]

                for s, v in schema_dict['NON_CAT_SLOTS_DICT'][typ][f['service']].items():
                    frames[serv]['non_cat_slots']['status'][v['slot_id']] = 0

                for s, v in schema_dict['CAT_SLOTS_DICT'][typ][f['service']].items():
                    frames[serv]['cat_slots'][v['slot_id']] = 0

                for (s, v) in f['turn_states']:
                    if s in schema_dict['CAT_SLOTS_DICT'][typ][f['service']].keys():
                        frames[serv]['cat_slots'][schema_dict['CAT_SLOTS_DICT'][typ][f['service']][s]['slot_id']] = schema_dict['CAT_SLOTS_DICT'][typ][f['service']][s]['values'][v[0]]
                    elif s in schema_dict['NON_CAT_SLOTS_DICT'][typ][f['service']].keys():
                        if v[0] == 'dontcare':
                            dc += 1
                            frames[serv]['non_cat_slots']['status'][schema_dict['NON_CAT_SLOTS_DICT'][typ][f['service']][s]['slot_id']] = 1

                        else:
                            ot+=1
                            frames[serv]['non_cat_slots']['status'][schema_dict['NON_CAT_SLOTS_DICT'][typ][f['service']][s]['slot_id']] = 2

                            utt_span_boundaries = _find_subword_indices({s:v}, turn.utterance, f['slots'], utterance_align, utterance_tok, 2+len(sys_res_tok))
                            if s in utt_span_boundaries:
                                frames[serv]['non_cat_slots']['start'][schema_dict['NON_CAT_SLOTS_DICT'][typ][f['service']][s]['slot_id']] = utt_span_boundaries[s][0]
                                frames[serv]['non_cat_slots']['end'][schema_dict['NON_CAT_SLOTS_DICT'][typ][f['service']][s]['slot_id']] = utt_span_boundaries[s][1]

                            elif f['service'] in turn.prev['frames']:
                                sys_res_span_boundaries = _find_subword_indices({s:v}, turn.prev['sys_res'], turn.prev['frames'][f['service']]['slots'], sys_res_align, sys_res_tok, 1)
                                if s in sys_res_span_boundaries:
                                    frames[serv]['non_cat_slots']['start'][schema_dict['NON_CAT_SLOTS_DICT'][typ][f['service']][s]['slot_id']] = sys_res_span_boundaries[s][0]
                                    frames[serv]['non_cat_slots']['end'][schema_dict['NON_CAT_SLOTS_DICT'][typ][f['service']][s]['slot_id']] = sys_res_span_boundaries[s][1]

                    else:
                        print('Slot name {} not found in schema.'.format(s))
                        exit()

            y_turn.append(copy.deepcopy(frames))
            id_turn.append([d_id, turn.t_id])
            inv_align_turn.append([sys_res_inv_align, utterance_inv_align])
            prev_sys_frame_turn.append(turn.prev)
    return id_turn, utt_true_turn, x_turn, x_turn_len, y_turn, inv_align_turn, prev_sys_frame_turn


def pad_dial(dials, pad=0):
    padded = []
    lens = [len(d) for d in dials]
    max_len = max(lens)

    for d, l in zip(dials, lens):
        seq = d + (max_len-l)*[pad]
        padded.append(seq)

    padded = torch.LongTensor(padded).to(config.DEVICE)
    att = (padded != pad).float().to(config.DEVICE)
    return padded, att, lens

def pad_frames(labels, schema_emb, typ='train'):
    n_cat_slots = schema_emb['CAT_SLOT_REPN'][typ].shape[-2]
    n_non_cat_slots = schema_emb['NON_CAT_SLOT_REPN'][typ].shape[-2]
    labels = [copy.deepcopy(f) for f in labels]
    max_serv = max([len(f) for f in labels])
    for i, frames in enumerate(labels):
        l = len(frames)
        labels[i].update({-c-1: {'active_intent': -1, 'req_slots': [], 'cat_slots': [-1]*n_cat_slots,  'non_cat_slots': {'status':[-1]*n_non_cat_slots, 'start':[-1]*n_non_cat_slots, 'end':[-1]*n_non_cat_slots}} for c in range(max_serv-l)})

    return labels

def get_batch(dial_id, utt_true_dial, X_dial, X_dial_len, Y_dial, inv_align, prev_sys_frame, tokenizer, schema_emb, batch_size=32, shuffle=True, typ='train'):
    PAD = tokenizer.convert_tokens_to_ids(['[PAD]'])[0]
    while True:
        if shuffle:
            data = list(zip(dial_id, utt_true_dial, X_dial, X_dial_len, Y_dial, inv_align, prev_sys_frame))
            random.shuffle(data)
            dial_id, utt_true_dial, X_dial, X_dial_len, Y_dial, inv_align, prev_sys_frame = zip(*data)

        for i in range(0, len(X_dial), batch_size):
            d_id_batch = dial_id[i:i+batch_size]
            utt_true_batch = utt_true_dial[i:i+batch_size]
            x_batch = X_dial[i:i+batch_size]
            x_batch_len = X_dial_len[i:i+batch_size]
            y_batch = Y_dial[i:i+batch_size]
            inv_align_batch = inv_align[i:i+batch_size]
            prev_sys_frame_batch = prev_sys_frame[i:i+batch_size]
            x_batch, att_mask, dial_lens = pad_dial(x_batch, pad=PAD)
            x_token_ids = np.zeros((x_batch.size()))
            for i, (a, b) in enumerate(x_batch_len):
                x_token_ids[i, a:(a+b)] = 1
            y_batch = pad_frames(y_batch, schema_emb, typ)
            x_token_ids = torch.from_numpy(x_token_ids).long().to(config.DEVICE)
            yield d_id_batch, utt_true_batch, x_batch, x_token_ids, att_mask, y_batch, dial_lens, inv_align_batch, prev_sys_frame_batch

def get_frame_level_data(service_id, y, schema_emb, typ='train'):
    true_labels, masking, possible = {}, {}, {}
    batch_size = len(y)

    possible['services'] = schema_emb['SERVICE_REPN'][typ][[s for s in service_id]]
    possible['services'] = torch.Tensor(possible['services']).to(config.DEVICE)

    #intents
    true_labels['intents'] = [y[i][service_id[i]]['active_intent'] for i in range(len(y))]
    true_labels['intents'] = torch.LongTensor(true_labels['intents']).to(config.DEVICE)
    # true_labels['intents'] ==> B

    possible['intents'] = schema_emb['INTENT_REPN'][typ][[s for s in service_id]]
    possible['intents'] = torch.Tensor(possible['intents']).to(config.DEVICE)
    possible['intents'] = possible['intents'].view(batch_size, possible['intents'].size(1), -1)
    # possible['intents'] ==> B x n_intents x dim

    masking['intents'] = schema_emb['INTENT_MASK'][typ][[s for s in service_id]]
    masking['intents'] = torch.Tensor(masking['intents']).to(config.DEVICE)
    masking['intents'] = masking['intents'].view(batch_size, -1)
    # masking['intents'] ==> B x n_intents

    #req slots
    possible['req_slots'] = schema_emb['SLOT_REPN'][typ][[s for s in service_id]]
    possible['req_slots'] = torch.Tensor(possible['req_slots']).to(config.DEVICE)
    possible['req_slots'] = possible['req_slots'].view(batch_size, possible['req_slots'].size(1), -1)

    masking['req_slots'] = schema_emb['SLOT_MASK'][typ][[s for s in service_id]]
    masking['req_slots'] = torch.Tensor(masking['req_slots']).to(config.DEVICE)
    masking['req_slots'] = masking['req_slots'].view(batch_size, -1)

    req_slots_true = [y[i][service_id[i]]['req_slots'] for i in range(len(y))]
    true_labels['req_slots'] = torch.zeros((batch_size, possible['req_slots'].size(1))).to(config.DEVICE)
    for i, d in enumerate(req_slots_true):
        if len(d) > 0:
            true_labels['req_slots'][i, d] = 1.

    #categorical slots
    possible['cat_slots'] = schema_emb['CAT_SLOT_REPN'][typ][[s for s in service_id]]
    possible['cat_slots'] = torch.Tensor(possible['cat_slots']).to(config.DEVICE)
    possible['cat_slots'] = possible['cat_slots'].view(batch_size, possible['cat_slots'].size(-2), possible['cat_slots'].size(-1))
    # possible['cat_slots'] ==> B x n_slots x dim

    masking['cat_slots'] = schema_emb['CAT_SLOT_MASK'][typ][[s for s in service_id]]
    masking['cat_slots'] = torch.Tensor(masking['cat_slots']).to(config.DEVICE)
    masking['cat_slots'] = masking['cat_slots'].view(batch_size, -1)
    # masking['cat_slots'] ==> B x n_slots

    #categorical values
    true_labels['cat_values'] = [y[i][service_id[i]]['cat_slots'] for i in range(len(y))]
    true_labels['cat_values'] = torch.LongTensor(true_labels['cat_values']).to(config.DEVICE)
    # true_labels['cat_values'] ==> B x n_slots

    possible['cat_values'] = schema_emb['CAT_VALUE_REPN'][typ][[s for s in service_id]]
    possible['cat_values'] = torch.Tensor(possible['cat_values']).to(config.DEVICE)
    possible['cat_values'] = possible['cat_values'].view(batch_size, possible['cat_values'].size(1), possible['cat_values'].size(2), -1)
    # possible['cat_values'] ==> B x n_slots x n_values x dim

    masking['cat_values'] = schema_emb['CAT_VALUE_MASK'][typ][[s for s in service_id]]
    masking['cat_values'] = torch.Tensor(masking['cat_values']).to(config.DEVICE)
    masking['cat_values'] = masking['cat_values'].view(batch_size, possible['cat_values'].size(1), -1)
    # masking['cat_values'] ==> B x n_slots x n_values

    #non_categorical slots
    possible['non_cat_slots'] = schema_emb['NON_CAT_SLOT_REPN'][typ][[s for s in service_id]]
    possible['non_cat_slots'] = torch.Tensor(possible['non_cat_slots']).to(config.DEVICE)
    possible['non_cat_slots'] = possible['non_cat_slots'].view(batch_size, possible['non_cat_slots'].size(-2), -1)
    # possible['non_cat_slots'] ==> B x n_slots x dim

    masking['non_cat_slots'] = schema_emb['NON_CAT_SLOT_MASK'][typ][[s for s in service_id]]
    masking['non_cat_slots'] = torch.Tensor(masking['non_cat_slots']).to(config.DEVICE)
    masking['non_cat_slots'] = masking['non_cat_slots'].view(batch_size, -1)
    # possible['non_cat_slots'] ==> B x n_slots

    #non_categorical status and values index
    true_labels['non_cat_values_status'] = [y[i][service_id[i]]['non_cat_slots']['status'] for i in range(len(y))]
    true_labels['non_cat_values_status'] = torch.LongTensor(true_labels['non_cat_values_status']).to(config.DEVICE)
    # true_labels['non_cat_values_status'] ==> B x n_slots

    true_labels['non_cat_values_start'] = [y[i][service_id[i]]['non_cat_slots']['start'] for i in range(len(y))]
    true_labels['non_cat_values_start'] = torch.LongTensor(true_labels['non_cat_values_start']).to(config.DEVICE)
    # true_labels['non_cat_values_start'] ==> B x n_slots

    true_labels['non_cat_values_end'] = [y[i][service_id[i]]['non_cat_slots']['end'] for i in range(len(y))]
    true_labels['non_cat_values_end'] = torch.LongTensor(true_labels['non_cat_values_end']).to(config.DEVICE)
    # true_labels['non_cat_values_end'] ==> B x n_slots

    #mask for padded services
    service_id = torch.LongTensor(service_id)
    masking['cat_values'][service_id<0] = 0.
    masking['cat_slots'][service_id<0] = 0.
    masking['non_cat_slots'][service_id<0] = 0.
    masking['req_slots'][service_id<0] = 0.

    possible['cat_slots'][service_id<0] = 0.
    possible['cat_values'][service_id<0] = 0.
    possible['non_cat_slots'][service_id<0] = 0.
    possible['req_slots'][service_id<0] = 0.

    return possible, masking, true_labels
