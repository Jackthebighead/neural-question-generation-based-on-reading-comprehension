import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

import json
from pathlib import Path
import torch

from transformers import BertModel

from preprocessing import Preprocess


# runtime environment
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# paths
squad_path = Path('dataset_squad/')
bert_path = Path('pretrained_models/bert')
model_path = Path('pretrained_models/bert/')

# encoder parameter
stage = 'feature_based'
bert_model = 'bert-base-cased'
print('Setup: downloading: ', bert_model)
BertModel.from_pretrained(bert_model).save_pretrained(model_path / stage / bert_model)
with (model_path / stage / bert_model / 'config.json').open('r') as f:
    conf = json.load(f)
    bert_hidden_size = conf['hidden_size']
    bert_vocab_size = conf['vocab_size']

#optimizer
weight_decay = 0.001
betas = (0.9, 0.999) # only for Adam
lr = 0.05
lr_adam = 2e-5  # lr for adam
momentum = 0.9 # only for SGD

# decoder parameter
decoder_hidden_size = 512
decoder_input_size = 512  # embedding dimesions
attention_hidden_size = 512
num_layers = 2
clip = 1
dropout = 0.5

# training parameters
epochs = 18
mb = 16
dl_workers = 0
checkpoint = None
pretrained_type = 'feature_based'  # 'feature_based', 'all','last_four'
checkpoint_path = 'checkpoints/'
