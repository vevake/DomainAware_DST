import os
import torch

DATA_DIR = 'data/'
OUT_DIR = 'results/'

BERT_MODEL = 'bert-base-cased'
BERT_DIR = 'Bert/'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DOMAIN = 'all' #['single', 'multi', 'all']
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 2e-5
GRAD_STEPS = 1