import json
import pandas as pd
import os

def get_schema(schemas):
    services = set()
    intents, slots, slot_type, values, slots_to_track, intent_slots = {}, {}, {}, {}, {}, {}
    service_desc, intent_desc, slot_desc = {}, {}, {}
    for sch in schemas:
        s_name = sch['service_name']
        services.update([s_name])
        service_desc[s_name] = sch['description']

        intents[s_name] = [s['name'] for s in sch['intents']]
        intent_desc[s_name] = {s['name']:s['description'] for s in sch['intents']}

        intent_slots[s_name] = {intnt:[a for s in sch['intents'] for a in s['required_slots']+list(s['optional_slots'].keys()) if s['name']==intnt] for intnt in intents[s_name]}
        slots_to_track[s_name] = [a for s in sch['intents'] for a in s['required_slots']+list(s['optional_slots'].keys())]

        slots[s_name] = [s['name'] for s in sch['slots']]
        slot_desc[s_name] = {s['name']:s['description'] for s in sch['slots']}
        slot_type[s_name] = {}
        for s in sch['slots']:
            if s['name'] in slots_to_track[s_name]:
                if s['is_categorical']:
                    slot_type[s_name][s['name']] = 'categorical'
                else:
                    slot_type[s_name][s['name']] = 'free_form'

        values[s_name] = {}
        for s in sch['slots']:
            if s['name'] in slot_type[s_name].keys():
                if slot_type[s_name][s['name']] == 'categorical':
                    values[s_name][s['name']] = s['possible_values']

    schema = {'services': services, 'service_desc': service_desc,
              'intents': intents, 'intent_desc': intent_desc, 'intent_slots':intent_slots,
              'slots': slots, 'slot_desc': slot_desc,
              'slot_type': slot_type, 'slots_to_track': slots_to_track,
              'values':values}

    return schema