import six
import re
import pickle
import collections
import torch
import config
def get_dicts():
    schema_types = ['train', 'dev', 'test']
    schema_dict = {'ALL_SERVICES_DICT': {}, 'CAT_SLOTS_DICT': {}, 'NON_CAT_SLOTS_DICT': {}}
    for s in schema_types:
        with open(config.OUT_DIR + s + '_schema_dict.pkl', 'rb') as f:
            t = pickle.load(f)
            schema_dict['ALL_SERVICES_DICT'][s] = t['ALL_SERVICES_DICT']
            schema_dict['CAT_SLOTS_DICT'][s] = t['CAT_SLOTS_DICT']
            schema_dict['NON_CAT_SLOTS_DICT'][s] = t['NON_CAT_SLOTS_DICT']
    return schema_dict

def get_schema_emb():
    schema_types = ['train', 'dev', 'test']
    emb_names = ['SERVICE_REPN', 'INTENT_REPN', 'INTENT_MASK',
                'SLOT_REPN', 'SLOT_MASK', 'CAT_VALUE_REPN', 'CAT_VALUE_MASK',
                'CAT_SLOT_REPN', 'CAT_SLOT_MASK', 'NON_CAT_SLOT_REPN', 'NON_CAT_SLOT_MASK']
    schema_emb = {k: {} for k in emb_names}
    for s in schema_types:
        SCHEMA_EMB = torch.load(config.OUT_DIR + s +'_schema_emb.pt')
        for k in emb_names:
            schema_emb[k][s] = SCHEMA_EMB[k]
    return schema_emb

def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def _naive_tokenize(s):
    seq_tok = [tok for tok in re.split(r"([^a-zA-Z0-9])", convert_to_unicode(s)) if tok]
    return seq_tok


def _tokenize(tokenizer, utterance):
    tokens = _naive_tokenize(utterance)
    # Filter out empty tokens and obtain aligned character index for each token.
    alignments = {}
    char_index = 0
    bert_tokens = []
    # These lists store inverse alignments to be used during inference.
    bert_tokens_start_chars = []
    bert_tokens_end_chars = []
    for token in tokens:
        if token.strip():
            subwords = tokenizer.tokenize(token)
            # Store the alignment for the index of starting character and the
            # inclusive ending character of the token.
            alignments[char_index] = len(bert_tokens)
            bert_tokens_start_chars.extend([char_index] * len(subwords))
            bert_tokens.extend(subwords)
            # The inclusive ending character index corresponding to the word.
            inclusive_char_end = char_index + len(token) - 1
            alignments[inclusive_char_end] = len(bert_tokens) - 1
            bert_tokens_end_chars.extend([inclusive_char_end] * len(subwords))
        char_index += len(token)
    inverse_alignments = list(
        zip(bert_tokens_start_chars, bert_tokens_end_chars))
    return bert_tokens, alignments, inverse_alignments

def _find_subword_indices(slot_values, utterance, char_slot_spans,
                          alignments, subwords, bias):
    """Find indices for subwords corresponding to slot values."""
    span_boundaries = {}
    for slot, values in slot_values.items():
      # Get all values present in the utterance for the specified slot.
        value_char_spans = {}
        for slot_span in char_slot_spans:
            if slot_span["slot"] == slot:
                value = utterance[slot_span["start"]:slot_span["exclusive_end"]]
                start_tok_idx = alignments[slot_span["start"]]
                end_tok_idx = alignments[slot_span["exclusive_end"] - 1]
                if 0 <= start_tok_idx < len(subwords):
                    end_tok_idx = min(end_tok_idx, len(subwords) - 1)
                    value_char_spans[value] = (start_tok_idx + bias, end_tok_idx + bias)
        for v in values:
            if v in value_char_spans:
                span_boundaries[slot] = value_char_spans[v]
                break
    return span_boundaries