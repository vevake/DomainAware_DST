import torch
import copy
from torch import nn
from data_reader.read_data import get_frame_level_data, get_files
import config
import numpy as np
import json
from evaluation import schema
from evaluation.eval import main as get_metrics
import collections
import os


def get_predicted_dialog(dialog, all_predictions, schemas, typ='dev'):
    # Overwrite the labels in the turn with the predictions from the model. For
    # test set, these labels are missing from the data and hence they are added.
    dialog_id = dialog["dialogue_id"]
    # The slot values tracked for each service.
    all_slot_values = collections.defaultdict(dict)
    for turn_idx, turn in enumerate(dialog["turns"]):
        if turn["speaker"] == "USER":
            user_utterance = turn["utterance"]
            system_utterance = (dialog["turns"][turn_idx - 1]["utterance"] if turn_idx else "")
            for frame in turn["frames"]:
                predictions = all_predictions[(dialog_id, turn_idx, frame["service"])]
                slot_values = all_slot_values[frame["service"]]
                service_schema = schemas.get_service_schema(frame["service"])
                # Remove the slot spans and state if present.
                frame.pop("slots", None)
                frame.pop("state", None)

                # The baseline model doesn't predict slot spans. Only state predictions
                # are added.
                state = {}

                # Add prediction for active intent. Offset is subtracted to account for
                active_intent_id = predictions["intent_status"] #vevake
                state["active_intent"] = (active_intent_id) #vevake

                # Add prediction for requested slots.
                state["requested_slots"] = predictions["requested_slots"]

                # Add prediction for user goal (slot values).
                # Categorical slots.
                for slot, value in predictions["slot_value"].items():
                    slot_values[slot] = value

                state["slot_values"] = {s: [v] for s, v in slot_values.items()}
                frame["state"] = state
    return dialog


def get_inverse_dict(schema_dict, typ='train'):
    CAT_SLOTS_DICT_INV = {}
    CAT_SLOTS_DICT_INV[typ] = {}
    for serv, sv in schema_dict['CAT_SLOTS_DICT'][typ].items():
        idx = schema_dict['ALL_SERVICES_DICT'][typ][serv]['serv_id']

        CAT_SLOTS_DICT_INV[typ][idx] = {'serv_name': serv}#{v: k for k, v in v.items()}
        CAT_SLOTS_DICT_INV[typ][idx].update({v['slot_id']:{'slot_name':k} for k, v in sv.items()})
        for sname, sval in sv.items():
            CAT_SLOTS_DICT_INV[typ][idx][sval['slot_id']].update({v:s for s,v in sval['values'].items()})

    NON_CAT_SLOTS_DICT_INV = {}
    NON_CAT_SLOTS_DICT_INV[typ] = {}
    for serv, sv in schema_dict['NON_CAT_SLOTS_DICT'][typ].items():
        idx = schema_dict['ALL_SERVICES_DICT'][typ][serv]['serv_id']

        NON_CAT_SLOTS_DICT_INV[typ][idx] = {'serv_name': serv}#{v: k for k, v in v.items()}
        NON_CAT_SLOTS_DICT_INV[typ][idx].update({v['slot_id']:{'slot_name':k} for k, v in sv.items()})

    ALL_SERVICES_DICT_INV = {}
    ALL_SERVICES_DICT_INV[typ] = {}
    for sname, sval in schema_dict['ALL_SERVICES_DICT'][typ].items():
        idx = sval['serv_id']
        ALL_SERVICES_DICT_INV[typ][idx] = {'serv_name': sname}
        ALL_SERVICES_DICT_INV[typ][idx].update({'intents': {v:k for k,v in sval['intents'].items()}})
        ALL_SERVICES_DICT_INV[typ][idx].update({'slots': {v:k for k,v in sval['slots'].items()}})

    return ALL_SERVICES_DICT_INV, CAT_SLOTS_DICT_INV, NON_CAT_SLOTS_DICT_INV

def evaluate(model, data_gen, length, schema_dict, schema_emb, typ='dev'):
    model.eval()
    st_true, st_pred = [], []
    all_predictions = {}
    ALL_SERVICES_DICT_INV, CAT_SLOTS_DICT_INV, NON_CAT_SLOTS_DICT_INV = get_inverse_dict(schema_dict, typ)
    for b_id in range(0, length):
        d_id, utt_true, x, x_len, attn_mask, y, dial_lens, inv_align, prev_sys_frame = next(iter(data_gen))
        with torch.no_grad():
            batch_services = [list(y[i].keys()) for i in range(len(y))]
            for frame in range(len(y[0])):
                service_id =  [batch_services[i][frame] for i in range(len(y))]
                possible, masking, true_labels = get_frame_level_data(service_id, y, schema_emb, typ=typ)

                scores = model(x, x_len, attn_mask, possible, masking)

                intent_score, req_slot_score, cat_status_score, cat_value_score, non_cat_status_score, non_cat_value_score = scores

                n_dials = x.size(0)
                n_tokens = x.size(1)

                n_cat_slots = possible['cat_slots'].size(-2)
                n_non_cat_slots = possible['non_cat_slots'].size(1)
                n_values = possible['cat_values'].size(-2)
                n_req_slots = possible['req_slots'].size(-2)

                pred = {}
                pred['intents'] = intent_score.topk(1)[1]
                pred['intents'] = pred['intents'].view(x.size(0), -1)

                pred['req_slots'] = nn.Sigmoid()(req_slot_score)
                pred['req_slots'] = pred['req_slots'] * masking['req_slots']

                pred['cat_status'] = cat_status_score.topk(1)[1]
                pred['cat_slots'] = cat_value_score.view(n_dials, n_cat_slots, n_values)
                pred['cat_slots'] = nn.Sigmoid()(pred['cat_slots'][:, :, 1:]) * masking['cat_values'][:, :, 1:].contiguous()
                pred['cat_slots'] = pred['cat_slots'].topk(1)

                # non categorical slot status
                pred['non_cat_status'] = non_cat_status_score.topk(1)[1]

                # non categorical slot value span
                attention_mask_non_cat = attn_mask.unsqueeze(1).repeat(1, n_non_cat_slots, 1).clone()
                start_span_score = nn.Softmax(dim=-1)(non_cat_value_score[:, :, :, 0].squeeze(-1))
                start_span_score[attention_mask_non_cat==0.] = 0.

                end_span_score = nn.Softmax(dim=-1)(non_cat_value_score[:, :, :, 1].squeeze(-1))
                end_span_score[attention_mask_non_cat==0.] = 0.
                # start_span_score --> B x n_non_cat_slots x n_tokens

                total_score = start_span_score.unsqueeze(3) + end_span_score.unsqueeze(2)
                # B x n_non_cat_slots x n_tokens x 1
                # B x n_non_cat_slots x 1 x n_tokens
                # total_score --> B x n_non_cat_slots x n_tokens x n_tokens

                start_idx = torch.arange(n_tokens).reshape(1,-1,1)
                end_idx = torch.arange(n_tokens).reshape(1,1,-1)
                valid_idx_mask = (start_idx <= end_idx).float().to(config.DEVICE).repeat(n_dials, n_non_cat_slots, 1, 1)
                # invalid_idx_mask --> B x n_non_cat_slots x n_tokens x n_tokens

                total_score = total_score * valid_idx_mask
                span_idx = total_score.reshape(n_dials, n_non_cat_slots, n_tokens**2)
                # span_idx --> B x n_non_cat_slots x (n_tokens*n_tokens)
                span_idx = span_idx.topk(1)[1].squeeze(-1).float()
                # span_idx --> B x n_non_cat_slots x 1

                pred['non_cat_value_start'] = (span_idx/n_tokens).floor()
                # pred['non_cat_value_start'] --> B x n_non_cat_slots x 1
                pred['non_cat_value_end'] = torch.fmod(span_idx, n_tokens)
                # pred['non_cat_value_end'] --> B x n_non_cat_slots x 1

                pred['cat_slots'] = (pred['cat_slots'][0].squeeze(-1), pred['cat_slots'][1].squeeze(-1))

                for i, d_len in enumerate(dial_lens):
                    if service_id[i] >= 0:

                        req_slot_pred = list(torch.nonzero(pred['req_slots'][i]>0.5).view(-1).cpu().numpy())
                        req_slot_pred = [ALL_SERVICES_DICT_INV[typ][service_id[i]]['slots'][p] for p in req_slot_pred]

                        cat_value = {}
                        non_cat_value = {}
                        for s in range(int(masking['cat_slots'][i].sum().item())):
                            slot_name = CAT_SLOTS_DICT_INV[typ][service_id[i]][s]['slot_name']

                            status = pred['cat_status'][i][s].item()
                            if status ==1:
                                cat_value[slot_name] = 'dontcare'

                            elif status == 2:
                                v = pred['cat_slots'][0][i][s].item()
                                if v > 0.5:
                                    cat_value[slot_name] = CAT_SLOTS_DICT_INV[typ][service_id[i]][s][pred['cat_slots'][1][i][s].item()+1]

                        for s in range(int(masking['non_cat_slots'][i].sum().item())):
                            v = pred['non_cat_status'][i][s].item()

                            slot_name = NON_CAT_SLOTS_DICT_INV[typ][service_id[i]][s]['slot_name']

                            if v == 1:
                                non_cat_value[slot_name] = 'dontcare'
                            elif v == 2:

                                start_idx = int(pred['non_cat_value_start'][i][s].item())
                                end_idx = int(pred['non_cat_value_end'][i][s].item())

                                sys_res = utt_true[i][0]
                                usr_utt = utt_true[i][1]

                                sys_res_len = len(inv_align[i][0])
                                utt_len = len(inv_align[i][1])
                                if start_idx > 0 and start_idx < (sys_res_len+1) and end_idx < sys_res_len+1:
                                    start_idx = start_idx - 1
                                    end_idx = end_idx - 1
                                    non_cat_value[slot_name] = sys_res[inv_align[i][0][start_idx][0]: inv_align[i][0][end_idx][1]+1]
                                elif start_idx >= (sys_res_len+1) and end_idx >= (sys_res_len+1) and end_idx < (sys_res_len+2+utt_len):
                                    start_idx = start_idx - (sys_res_len+2)
                                    end_idx = end_idx - (sys_res_len+2)
                                    non_cat_value[slot_name] = usr_utt[inv_align[i][1][start_idx][0]: inv_align[i][1][end_idx][1]+1]

                        all_slots = {}
                        for s, v in cat_value.items():
                            all_slots[s] = v

                        for s, v in non_cat_value.items():
                            all_slots[s] = v

                        all_predictions[(d_id[i][0], d_id[i][1],
                                         ALL_SERVICES_DICT_INV[typ][service_id[i]]['serv_name'])] = {'intent_status': ALL_SERVICES_DICT_INV[typ][service_id[i]]['intents'][pred['intents'][i].item()],
                                                                                                    'requested_slots': req_slot_pred,
                                                                                                    'slot_value': copy.deepcopy(all_slots)}

    model.train()

    schema_json_file = os.path.join(config.DATA_DIR, typ, "schema.json")
    schemas = schema.Schema(schema_json_file)
    prediction_dir = os.path.join(config.OUT_DIR, 'pred')

    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)

    for f in os.listdir(prediction_dir):
        os.remove(os.path.join(prediction_dir,f))

    files = get_files(typ)

    for f in files:
        dialogues = json.load(open(f, 'r'))
        pred_dialogues= []
        for d in dialogues:
            pred_dialogues.append(get_predicted_dialog(d, all_predictions, schemas, typ))

        output_file_path = os.path.join(prediction_dir, os.path.basename(f))
        with open(output_file_path, 'w') as f2:
            json.dump(pred_dialogues, f2, indent=2, separators=(",", ": "), sort_keys=True)


    dstc8_data_dir = os.path.join(config.DATA_DIR, typ)
    eval_set = typ
    output_metric_file = prediction_dir + 'result.txt'
    metrics = get_metrics(prediction_dir, config.DATA_DIR, eval_set, output_metric_file)
    print(f'{typ} results')
    print({k: np.round(v, 3) for k, v in metrics.items()})
    return all_predictions, metrics['joint_goal_accuracy'], metrics['average_goal_accuracy']