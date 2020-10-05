import json
import os
from data_reader.schema_reader import get_schema
import config
import pickle

def get_schema_dict(schema):
    ALL_SERVICES = list(schema['services'])
    ALL_SERVICES.sort()
    ALL_SERVICES_DICT, ALL_SERVICES_DICT_CAT, ALL_SERVICES_DICT_NONCAT = {}, {}, {}
    for i, serv in enumerate(ALL_SERVICES):
        ALL_SERVICES_DICT[serv] = {'serv_id': i}
        ALL_SERVICES_DICT[serv].update({'serv_desc': schema['service_desc'][serv]})
        ALL_SERVICES_DICT[serv].update({'intents': {'NONE': 0}, 'intents_desc': {'NONE': ''}})

        ALL_SERVICES_DICT_CAT[serv] = {'serv_id': i}
        ALL_SERVICES_DICT_CAT[serv].update({'serv_desc': schema['service_desc'][serv]})

        ALL_SERVICES_DICT_NONCAT[serv] = {'serv_id': i}
        ALL_SERVICES_DICT_NONCAT[serv].update({'serv_desc': schema['service_desc'][serv]})

    for serv, intents in schema['intents'].items():
        ALL_SERVICES_DICT[serv]['intents'].update({intnt:i+1 for i, intnt in enumerate(intents)})
        ALL_SERVICES_DICT[serv]['intents_desc'].update({intnt:desc  for i, (intnt, desc) in enumerate(schema['intent_desc'][serv].items())})

    for serv, slots in schema['slots'].items():
        ALL_SERVICES_DICT[serv]['slots'] = {s:i for i, s in enumerate(slots)}
        ALL_SERVICES_DICT[serv].update({'slots_desc': {s:desc  for i, (s, desc) in enumerate(schema['slot_desc'][serv].items())}})

        ALL_SERVICES_DICT_CAT[serv]['slots'] = {}
        ALL_SERVICES_DICT_NONCAT[serv]['slots'] = {}
        for i, s in enumerate(slots):
            if s in schema['slots_to_track'][serv]:
                if schema['slot_type'][serv][s] == 'categorical':
                    ALL_SERVICES_DICT_CAT[serv]['slots'].update({s:len(ALL_SERVICES_DICT_CAT[serv]['slots'])})
                else:
                    ALL_SERVICES_DICT_NONCAT[serv]['slots'].update({s:len(ALL_SERVICES_DICT_NONCAT[serv]['slots'])})

        ALL_SERVICES_DICT_CAT[serv].update({'slots_desc': {s:desc  for i, (s, desc) in enumerate(schema['slot_desc'][serv].items()) if s in ALL_SERVICES_DICT_CAT[serv]['slots'].keys()}})
        ALL_SERVICES_DICT_NONCAT[serv].update({'slots_desc': {s:desc  for i, (s, desc) in enumerate(schema['slot_desc'][serv].items()) if s in ALL_SERVICES_DICT_NONCAT[serv]['slots'].keys()}})


    CAT_SLOTS_DICT, NON_CAT_SLOTS_DICT = {}, {}
    for serv, slots in schema['slot_type'].items():
        cat, non_cat = [], []
        for k, v in slots.items():
            if v=='categorical':
                cat.append(k)
            else:
                non_cat.append(k)

        NON_CAT_SLOTS_DICT[serv] = {s: {'slot_id': ALL_SERVICES_DICT_NONCAT[serv]['slots'][s]} for i, s in enumerate(non_cat)}
        CAT_SLOTS_DICT[serv] = {s: {'slot_id': ALL_SERVICES_DICT_CAT[serv]['slots'][s]} for i, s in enumerate(cat)}

    for serv, slots in schema['values'].items():
        for k, v in slots.items():
            CAT_SLOTS_DICT[serv][k]['values'] = {'NONE': 0, 'dontcare': 1}
            CAT_SLOTS_DICT[serv][k]['values'].update({a:i+2 for i, a in enumerate(v)})

    return ALL_SERVICES_DICT, ALL_SERVICES_DICT_CAT, ALL_SERVICES_DICT_NONCAT, CAT_SLOTS_DICT, NON_CAT_SLOTS_DICT

if __name__ == "__main__":

    schemas_to_create = ['train', 'dev', 'test']

    if not os.path.exists(config.OUT_DIR):
        os.makedirs(config.OUT_DIR)

    for s in schemas_to_create:
        schema = json.load(open(os.path.join(config.DATA_DIR, s, 'schema.json'), 'r'))
        schema = get_schema(schema)

        ALL_SERVICES_DICT, ALL_SERVICES_DICT_CAT, ALL_SERVICES_DICT_NONCAT, CAT_SLOTS_DICT, NON_CAT_SLOTS_DICT = get_schema_dict(schema)

        d = {'ALL_SERVICES_DICT': ALL_SERVICES_DICT,
        'ALL_SERVICES_DICT_CAT': ALL_SERVICES_DICT_CAT,
        'ALL_SERVICES_DICT_NONCAT': ALL_SERVICES_DICT_NONCAT,
        'CAT_SLOTS_DICT': CAT_SLOTS_DICT,
        'NON_CAT_SLOTS_DICT': NON_CAT_SLOTS_DICT,
        'INTENT_SLOTS': schema['intent_slots'],
        }
        with open(config.OUT_DIR + s + '_schema_dict.pkl', 'wb') as f:
            pickle.dump(d, f)