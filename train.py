from data_reader.utils import get_dicts, get_schema_emb
from data_reader.read_data import get_dialogues, get_seqs, get_batch, get_frame_level_data
from models.utils import get_bert_tokenizer
from models.model import Model
import numpy as np
from loss import LossFn
from transformers import *
import torch
import os
from tqdm import tqdm
import config
from evaluate import evaluate

def main(train, dev, test):
    d_id_train, utt_true_train, x_train, x_train_len, y_train, inv_align_train, prev_sys_frame_train = train
    d_id_dev, utt_true_dev, x_dev, x_dev_len, y_dev, inv_align_dev, prev_sys_frame_dev = dev
    d_id_test, utt_true_test, x_test, x_test_len, y_test, inv_align_test, prev_sys_frame_test = test

    model = Model(bert_dir=os.path.join(config.BERT_DIR, config.BERT_MODEL), device=config.DEVICE)
    model.to(config.DEVICE)

    n_iterations = int(np.ceil(len(x_train) / config.BATCH_SIZE))
    num_total_steps = int(n_iterations // config.GRAD_STEPS * config.BATCH_SIZE)
    warmup_proportion = 0.1
    max_grad_norm = 1.0
    num_warmup_steps = int(num_total_steps * warmup_proportion)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.LEARNING_RATE, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_total_steps)


    train_gen = get_batch(d_id_train, utt_true_train, x_train, x_train_len, y_train, inv_align_train, prev_sys_frame_train, tokenizer, schema_emb, batch_size=config.BATCH_SIZE, shuffle=True, typ='train')
    dev_gen = get_batch(d_id_dev, utt_true_dev, x_dev, x_dev_len, y_dev, inv_align_dev, prev_sys_frame_dev, tokenizer, schema_emb, batch_size=config.BATCH_SIZE, shuffle=False, typ='dev')
    test_gen = get_batch(d_id_test, utt_true_test, x_test, x_test_len, y_test, inv_align_test, prev_sys_frame_test, tokenizer, schema_emb, batch_size=config.BATCH_SIZE, shuffle=False, typ='test')
    best_dev_acc = 0.
    lossfn = LossFn()
    for epoch in range(config.EPOCHS):
        total_loss, intent_loss, req_slot_loss, cat_value_loss, cat_status_loss, non_cat_status_loss, non_cat_start_span_loss, non_cat_end_span_loss, cnt = 0., 0., 0., 0., 0., 0., 0., 0., 0
        pbar = tqdm(range(n_iterations))
        model.train()
        print('\nEpoch :{}'.format(epoch+1))
        for b_id in pbar:
            d_id, utt_true, x, x_len, attn_mask, y, dial_lens, inv_align, prev_sys_frame = next(iter(train_gen))
            batch_services = [list(y[i].keys()) for i in range(len(y))]
            for frame in range(len(y[0])):
                service_id =  [batch_services[i][frame] for i in range(len(y))]
                possible, masking, true_labels = get_frame_level_data(service_id, y, schema_emb)

                scores = model(x, x_len, attn_mask, possible, masking)

                intent_score, req_slot_score, cat_status_score, cat_value_score, non_cat_status_score, non_cat_value_score = scores

                loss = {}
                loss['intents'] = lossfn.get_intent_loss(intent_score.view(-1, intent_score.size(-1)), true_labels['intents'].view(-1))
                loss['req_slots'] = lossfn.get_req_slot_loss(req_slot_score, true_labels['req_slots'], masking['req_slots'])

                n_dials = x.size(0)
                n_tokens = x.size(1)

                n_cat_slots = possible['cat_slots'].size(-2)
                n_non_cat_slots = possible['non_cat_slots'].size(-2)
                n_values = possible['cat_values'].size(-2)
                n_req_slots = possible['req_slots'].size(-2)

                cat_values_tmp = torch.zeros((n_dials, n_cat_slots, n_values-1)).to(config.DEVICE)
                for i, d in enumerate(true_labels['cat_values']):
                    for j, t in enumerate(d):
                        if t > 0:
                            cat_values_tmp[i, j, t-1] = 1.
                loss['cat_slots'] = lossfn.get_cat_value_loss(cat_value_score[:, :, 1:].contiguous().view(-1), cat_values_tmp.view(-1), masking['cat_values'][:, :, 1:].contiguous().view(-1))

                true_labels['cat_values'][true_labels['cat_values']>1] = 2
                loss['cat_status'] = lossfn.get_status_loss(cat_status_score.view(-1, 3), true_labels['cat_values'].view(-1), masking['cat_slots'])

                loss['non_cat_status'] = lossfn.get_status_loss(non_cat_status_score.view(-1, 3), true_labels['non_cat_values_status'].view(-1), masking['non_cat_slots'])
                loss['non_cat_start_span'] = lossfn.get_noncat_value_loss(non_cat_value_score[:, :, :, 0].view(-1, n_tokens), true_labels['non_cat_values_start'].view(-1))
                loss['non_cat_end_span'] = lossfn.get_noncat_value_loss(non_cat_value_score[:, :, :, 1].view(-1, n_tokens), true_labels['non_cat_values_end'].view(-1))

                intent_loss += loss['intents'].item()
                req_slot_loss += loss['req_slots'].item()
                cat_value_loss += loss['cat_slots'].item()
                cat_status_loss += loss['cat_status'].item()
                non_cat_status_loss += loss['non_cat_status'].item()
                non_cat_start_span_loss += loss['non_cat_start_span'].item()
                non_cat_end_span_loss += loss['non_cat_end_span'].item()

                loss = sum(loss.values())
                total_loss += loss.item()
                cnt += 1

                grad = loss / config.GRAD_STEPS
                grad.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            pbar.set_description('TL:{:.4f}, IL:{:.4f}, RL:{:.4f}, CS:{:.4f}, CL:{:.4f}, NSL:{:.4f}, NSP:{:.4f}, NEP:{:.4f}'.format(total_loss/cnt, intent_loss/cnt, req_slot_loss/cnt, cat_status_loss/cnt, cat_value_loss/cnt, non_cat_status_loss/cnt, non_cat_start_span_loss/cnt, non_cat_end_span_loss/cnt))

        print('Evauating...')
        dev_pred, jnt_gl, avg_gl = evaluate(model, dev_gen, length=int(np.ceil(len(x_dev)/config.BATCH_SIZE)), schema_dict=schema_dict, schema_emb=schema_emb)
        if avg_gl > best_dev_acc:
            # test_pred, jnt_gl_test, avg_gl_test = evaluate(model, test_gen, length=int(np.ceil(len(x_test)/config.BATCH_SIZE)), schema_dict=schema_dict, schema_emb=schema_emb, typ='test')
            torch.save(model.state_dict(), config.OUT_DIR + 'model.pt')
            best_dev_acc = avg_gl

    model.load_state_dict(torch.load(config.OUT_DIR + 'model.pt'))
    test_pred, jnt_gl_test, avg_gl_test = evaluate(model, test_gen, length=int(np.ceil(len(x_test)/config.BATCH_SIZE)), schema_dict=schema_dict, schema_emb=schema_emb, typ='test')

if __name__ == "__main__":
    schema_dict = get_dicts()
    schema_emb = get_schema_emb()

    train_dialogues, dev_dialogues, test_dialogues = get_dialogues()
    tokenizer = get_bert_tokenizer()

    train = get_seqs(train_dialogues, tokenizer, schema_dict, schema_emb, typ='train')
    dev = get_seqs(dev_dialogues, tokenizer, schema_dict, schema_emb, typ='dev')
    test = get_seqs(test_dialogues, tokenizer, schema_dict, schema_emb, typ='test')

    main(train, dev, test)
