from data_reader.utils import get_dicts, get_schema_emb
from data_reader.read_data import get_dialogues, get_seqs, get_batch
from models.utils import get_bert_tokenizer
from models.model import Model
import numpy as np
from transformers import *
import torch
import os
import config
from evaluate import evaluate

def main(test):
    d_id_test, utt_true_test, x_test, x_test_len, y_test, inv_align_test, prev_sys_frame_test = test

    model = Model(bert_dir=os.path.join(config.BERT_DIR, config.BERT_MODEL), device=config.DEVICE)
    model.to(config.DEVICE)
    model.load_state_dict(torch.load(config.OUT_DIR + 'model.pt'))

    test_gen = get_batch(d_id_test, utt_true_test, x_test, x_test_len, y_test, inv_align_test, prev_sys_frame_test,
                            tokenizer, schema_emb, batch_size=config.BATCH_SIZE, shuffle=False, typ='test')

    test_pred, jnt_gl_test, avg_gl_test = evaluate(model, test_gen, length=int(np.ceil(len(x_test)/config.BATCH_SIZE)),
                                                    schema_dict=schema_dict, schema_emb=schema_emb, typ='test')

if __name__ == "__main__":
    schema_dict = get_dicts()
    schema_emb = get_schema_emb()

    train_dialogues, dev_dialogues, test_dialogues = get_dialogues()
    tokenizer = get_bert_tokenizer()

    test = get_seqs(test_dialogues, tokenizer, schema_dict, schema_emb, typ='test')

    main(test)
