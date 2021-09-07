

import logging
logging.getLogger('transformers').setLevel(logging.WARNING)
log = logging.getLogger(__name__)

import time
import math
import os

import torch
from torch import optim, nn, cuda
from transformers import AdamW
from torch.utils.data import DataLoader
from transformers import BertModel

from config import checkpoint, bert_path, mb, dl_workers, device, bert_hidden_size, decoder_hidden_size, bert_vocab_size, decoder_input_size, dropout, epochs, clip, model_path, stage, bert_model, pretrained_type, attention_hidden_size, num_layers, weight_decay, betas, lr, lr_adam, momentum, checkpoint_path
from model.utils import load_checkpoint, init_weights, save_checkpoint, enable_reproducibility, model_size, no_grad
from model import Attention, Decoder, Seq2Seq
from preprocessing import BertDataset
from run import train, eval
from run.utils.time import epoch_time

import logging

import torch
from nltk.translate.bleu_score import SmoothingFunction
from torch import nn
from nltk.translate import bleu
from transformers import BertTokenizer

pw_criterion = nn.CrossEntropyLoss(ignore_index=0)  # Pad Index
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

def test(model, device, dataloader, criterion):
    
    model.eval()

    epoch_loss = 0
    epoch_bleu = 0

    with torch.no_grad():
        
        predictions, labels = [], []

        for i, (input_, output_) in enumerate(dataloader):

            input_data, input_length = input_
            output_data, output_length = output_

            prediction = model([x.to(device) for x in input_data], output_data.to(device), 0)  

            sample_t = tokenizer.convert_ids_to_tokens(output_data[0].tolist())
            sample_p = tokenizer.convert_ids_to_tokens(prediction[0].max(1)[1].tolist())
            idx1 = sample_t.index('[PAD]') if '[PAD]' in sample_t else len(sample_t)
            idx2 = sample_p.index('[SEP]') if '[SEP]' in sample_p else len(sample_p)

            bleu, preds, lbls = bleu_score(prediction, output_data.to(device))
            predictions.extend(preds)
            labels.extend(lbls)

            trg_sent_len = prediction.size(1)

            prediction = prediction[:, 1:].contiguous().view(-1, prediction.shape[-1])
            output_data = output_data[:, 1:].contiguous().view(-1)  # Find a way to avoid calling contiguous

            
            pw_loss = pw_criterion(prediction, output_data.to(device))

            loss = criterion(prediction, output_data.to(device))
            loss = loss.view(-1, trg_sent_len - 1)
            loss = loss.sum(1)
            loss = loss.mean(0)

            if i % int(len(dataloader) * 0.1) == int(len(dataloader) * 0.1) - 1:
                log.info(f'Batch {i} Sentence loss: {loss.item()} Word loss: {pw_loss.item()} BLEU score: {bleu}\n'
                         f'Target {sample_t[1:idx1-1]}\n'
                         f'Prediction {sample_p[1:idx2-1]}\n\n')
                print('Batch:',i,'Sentence loss:',loss.item(),'Word loss:',pw_loss.item(),'BLEU score:',bleu)

            epoch_loss += pw_loss.item()
            epoch_bleu += bleu

        return epoch_loss / len(dataloader), epoch_bleu / len(dataloader), predictions, labels

def bleu_score(prediction, ground_truth):
    prediction = prediction.max(2)[1]
    acc_bleu = 0
    
    preds, labels = [], []

    for x, y in zip(ground_truth, prediction):
        x = tokenizer.convert_ids_to_tokens(x.tolist())
        y = tokenizer.convert_ids_to_tokens(y.tolist())
        idx1 = x.index('[PAD]') if '[PAD]' in x else len(x)
        idx2 = y.index('[SEP]') if '[SEP]' in y else len(y)
        
        print(' '.join(x[1:idx1 - 5]))
        print(' '.join(y[1:idx2 - 5]))
        
        pred = ' '.join(x[1:idx1 - 5])
        label = ' '.join(y[1:idx2 - 5])
        
        preds.append(pred)
        labels.append(label)

        acc_bleu += bleu([x[1:idx1 - 1]], y[1:idx2 - 1], smoothing_function=SmoothingFunction().method4)
    return acc_bleu / prediction.size(0), preds, labels


if __name__=='__main__':

    log = logging.getLogger(__name__)
    log.info(f'Running on device {cuda.current_device()}')
    print('Running on device: ', cuda.current_device())

    enable_reproducibility(1234)
    
    print('* Loading processed dataset...')
    train_set = BertDataset('dataset_squad/train_data.json')
    valid_set = BertDataset('dataset_squad/valid_data.json')
    
    print('* Loading dataset to torch...')
    training_loader = DataLoader(train_set, batch_size=mb, shuffle=True,
                                 num_workers=dl_workers, pin_memory=True if device == 'cuda' else False)
    valid_loader = DataLoader(valid_set, batch_size=mb, shuffle=True,
                              num_workers=dl_workers, pin_memory=True if device == 'cuda' else False)

    print('* Defining model...')
    attention = Attention(bert_hidden_size, decoder_hidden_size, attention_hidden_size)  # add attention_hidden_size
    decoder = Decoder(bert_vocab_size, decoder_input_size, bert_hidden_size, decoder_hidden_size, num_layers,
                      dropout, attention, device)
    encoder = BertModel.from_pretrained(model_path / stage / bert_model)

    model = Seq2Seq(encoder, decoder, device)
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')  # Pad Index
    
    # Checkpoint
    if checkpoint is None:
        checkpoint = 'checkpoints/model_epoch_16'
    
    print('\t pretrained_type: ', pretrained_type)
    unfreeze_layers = ['layer.10', 'layer.11','bert.pooler','out.']
    for name, param in model.encoder.named_parameters():
        if pretrained_type == 'all': break
        param.requires_grad = False
        if pretrained_type == 'last_four':
            for layer in unfreeze_layers:
                if layer in name:
                    param.requires_grad = True
                    break

    print('* Unfreezed last four layers of BERT Encoder.')

    if checkpoint is not None:
        print('* Load model from checkpoint:', checkpoint)
        last_epoch, model_dict, optim_dict, valid_loss_list, train_loss_list = load_checkpoint(checkpoint)
        
        model.load_state_dict(model_dict)

        print('* Using Checkpoint')
    else:
        print('* Wrong! Aint using checkpoint now!')
        last_epoch = 0
        valid_loss_list, train_loss_list = [], []
        model.apply(init_weights)

    model.to(device)
    
    print('* Start Validating...')
    
    start_time = time.time()
        
    valid_loss, bleu_score, preds, labels = test(model, device, valid_loader, criterion)
    
    print('* In the validation, the valid loss is:', valid_loss, 'And the blue_score is:', bleu_score)
    
    print('saving test results...')
    with open(self.pred_path, 'w') as f:
        f.write("\n".join(preds))
    with open(self.label_path, 'w') as f:
        f.write("\n".join(labels)) 
        
    print('start nlg-eval...')
    from nlgeval import compute_metrics, compute_individual_metrics
    metrics_dict = compute_metrics(hypothesis=self.pred_path, references=[self.label_path])
    
    print('\t Validation finished')
    