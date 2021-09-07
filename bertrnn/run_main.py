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

# best_valid_loss = float('inf')


# Importante! Se il training viene fermato e poi ripreso senza cambiare il seed lo shuffling non avviene

if __name__ == '__main__':
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
    
    # freeze layers          
    # -1 means feature_based tuning, >0 means only unfreeze some layers, =0 means unfreeze all layers(fine-tuning)
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
    
    # check params requires grad
    print('\t * All trainable parameters in Encoder:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print('\t\t', name)

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), weight_decay=weight_decay, lr=lr, momentum=momentum)
    #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), weight_decay=weight_decay, lr=lr_adam)
    # optimizer = optim.SGD(decoder.parameters(), weight_decay=weight_decay, lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')  # Pad Index

    if checkpoint is not None:
        last_epoch, model_dict, optim_dict, valid_loss_list, train_loss_list = load_checkpoint(checkpoint)
        last_epoch += 1
        model.load_state_dict(model_dict)
        best_valid_loss = min(valid_loss_list)

        optimizer.load_state_dict(optim_dict)
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

        print(f'Using Checkpoint')
    else:
        last_epoch = 0
        valid_loss_list, train_loss_list = [], []
        model.apply(init_weights)

    model.to(device)

    print('* Start training...')
    
    best_valid_loss = 100000000000
    for epoch in range(last_epoch, epochs):
        start_time = time.time()

        print('\t Training Epoch:', epoch+1)
        #log.info(f'Epoch {epoch+1} training')
        #train_loss = train(model, device, training_loader, optimizer, criterion, clip)

        print('\t Validation Epoch:', epoch+1)
        log.info(f'\nEpoch {epoch + 1} validation')
        valid_loss, bleu_score, preds, labels = eval(model, device, valid_loader, criterion)
        
        
        # nlg_eval
        print('saving test results...')
        with open('inferences.txt', 'w') as f:
            f.write("\n".join(preds))
        with open('labels.txt', 'w') as f:
            f.write("\n".join(labels)) 
        
        print('start nlg-eval...')
        from nlgeval import compute_metrics, compute_individual_metrics
        metrics_dict = compute_metrics(hypothesis='inferences.txt', references=['labels.txt'])

        print('\t Training and Validation finished')

        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            print('\t Current loss is better!')
            best_valid_loss = valid_loss
            print('\t Model checkpoint saved at: ', os.path.join(checkpoint_path,f'model_epoch_{epoch}'))
            save_checkpoint(os.path.join(checkpoint_path,f'model_epoch_{epoch}'), epoch, model, optimizer, valid_loss_list, train_loss_list)
            print('\t Model checkpoint saved at: ', checkpoint_path)
        
        log.info(f'\nEpoch: {epoch + 1:02} completed | Time: {epoch_mins}m {epoch_secs}s')
        log.info(f'\t Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        log.info(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f} | Val. BLEU {bleu_score}')
        print('\t\t Epoch completed | Time cost: ', time.time()-start_time)
        print('\t\t Train Loss: {:.3f} | Train PPL: {:7.3f}'.format(train_loss, math.exp(train_loss)))
        print('\t\t Validation Loss: {:.3f} |  Validation PPL: {:7.3f} | Validation BLEU {}'.format(valid_loss, math.exp(valid_loss), bleu_score))








