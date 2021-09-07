# coding utf-8
"""
Created in June 2021
@author: YUAN Yanzhe
"""

# Import Packages
import os
import nlp
import torch
from torch import nn
import logging
import datasets
from datasets import Dataset
from dataclasses import dataclass, field
from typing import Any, Union, Dict, List, Optional
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import Trainer as HFTrainer
from transformers import TrainingArguments
from transformers import HfArgumentParser


# Hyperparameters
logger = logging.getLogger(__name__)
device = 'cuda' if torch.cuda.is_available else 'cpu'
print('* You are using device: ', device)

max_source_length = 512
max_target_length = 32
pretrained_model = 'facebook/bart-base'  # "facebook/bart-base"
num_epochs = 4
evaluation_strategy = 'steps'  # epoch/no/steps
eval_steps = 5000
save_steps = 5000

print('* Your hyperparameter settings:')
print('\t PRETRAINED_MODEL:', pretrained_model)
print('\t MAX_SOURCE_LENGTH:', max_source_length)
print('\t MAX_TARGET_LENGTH:', max_target_length)
print('\t NUM_EPOCHS:', num_epochs)
print('\t EVALUATION_STRATEGY:', evaluation_strategy)
print('\t SAVE_STEPS:', save_steps)

# Functions    
def convert_to_features(example_batch):
    source_encoding = tokenizer.batch_encode_plus(
        example_batch['source_text'],
        max_length=max_source_length,
        padding='max_length',
        pad_to_max_length=True,
        truncation=True, 
    )
    target_encoding = tokenizer.batch_encode_plus(
        example_batch['target_text'],
        max_length=max_target_length,
        padding='max_length',
        pad_to_max_length=True,
        truncation=True, 
    )
    encodings = {
        'source_ids': source_encoding['input_ids'], 
        'target_ids': target_encoding['input_ids'],
        'attention_mask': source_encoding['attention_mask'],
    }

    return encodings


def get_correct_alignement(context, answer):  
    gold_text = answer['text'][0]
    start_idx = answer['answer_start'][0]
    end_idx = start_idx + len(gold_text)
    if context[start_idx:end_idx] == gold_text:
        return start_idx, end_idx       
    elif context[start_idx-1:end_idx-1] == gold_text:
        return start_idx-1, end_idx-1  
    elif context[start_idx-2:end_idx-2] == gold_text:
        return start_idx-2, end_idx-2 
    else:
        raise ValueError()
        
        
def process_qg_text(example):
    context = example['context']
    question = example['question']
    answer = example['answers']
    answer_text = answer['text'][0]
    start_pos, end_pos = get_correct_alignement(context, answer)
    qg_input = f"{context[:start_pos]} {hl} {answer_text} {hl} {context[end_pos:]}"
    qg_target = f"{question}"
    qg_input = qg_input + " </s>"
    qg_target = qg_target + " </s>"
    return {"source_text": qg_input, "target_text": qg_target}


def trim_batch(input_ids, pad_token_id, attention_mask=None,):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])

    
def get_predictions(model, tokenizer, data_loader, num_beams=4, max_length=32, length_penalty=1):
    model.to(device)
    
    predictions = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader):
            outs = model.generate(
                input_ids=batch['input_ids'].to(device), 
                attention_mask=batch['attention_mask'].to(device),
                num_beams=num_beams,
                max_length=max_length,
                length_penalty=length_penalty,
            )

            prediction = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
            predictions.extend(prediction)

    return predictions


# Customize Data
class T2TDataCollator():
    def __init__(self, tokenizer, model_type='bart', mode='training'):
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.mode = mode

    def __call__(self, batch: List) -> Dict[str, torch.Tensor]:

        input_ids = torch.stack([example['source_ids'] for example in batch])
        target_ids = torch.stack([example['target_ids'] for example in batch])
        attention_mask = torch.stack([example['attention_mask'] for example in batch])

        pad_token_id = self.tokenizer.pad_token_id
        
        input_ids, attention_mask = trim_batch(input_ids, pad_token_id, attention_mask=attention_mask)
        target_ids = trim_batch(target_ids, pad_token_id)
        
        if self.model_type == 'bart':
            decoder_input_ids = target_ids[:, :-1].contiguous()
            lm_labels = target_ids[:, 1:].clone()
            if self.mode == 'training':
                lm_labels[target_ids[:, 1:] == pad_token_id] = -100   
        else:  # self.model_type == 't5'
            print('You are using model_type: t5.')
            lm_labels = target_ids.clone()
            decoder_input_ids = self._shift_right(lm_labels)
            if self.mode == 'training':
                lm_labels[lm_labels[:, :] == pad_token_id] = -100
                
        params = {
            "input_ids": input_ids, 
            "attention_mask": attention_mask,
            "labels": lm_labels,
            "decoder_input_ids": decoder_input_ids
        }
        
        return params
    
    def _shift_right(self, input_ids):
        decoder_start_token_id = self.tokenizer.pad_token_id
        pad_token_id = self.tokenizer.pad_token_id

        assert (
            decoder_start_token_id is not None
        ), "<decoder_start_token_id> has to be defined."

        # shift the inputs to the right 
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        assert pad_token_id is not None, "<pad_token_id> has to be defined."
        
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        assert torch.all(shifted_input_ids >= 0).item(), "<labels> has only positive values and remove 100 backwards"

        return shifted_input_ids

    
# Define Trainer    
class Trainer(HFTrainer):
    def __init__(self, label_smoothing: float = 0, **kwargs):
        super().__init__(**kwargs)
        self.label_smoothing = label_smoothing
    
    # override to support label smoothing
    def _training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], optimizer: torch.optim.Optimizer
    ) -> float:
        model.train()
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.args.device)


        if isinstance(model, nn.DataParallel):
            inputs["return_tuple"] = True

        if self.label_smoothing == 0:
            outputs = model(**inputs)
            loss = outputs[0]
        else:
            labels = inputs.pop('labels')
            labels[labels == -100] = model.config.pad_token_id
            outputs = model(**inputs)
            lprobs = torch.nn.functional.log_softmax(outputs[0], dim=-1)
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs, labels, self.label_smoothing, ignore_index=model.config.pad_token_id
            )

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        else:
            loss.backward()

        return loss.item()


# Load Model and Datasets
print('* Loading Bart Tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
tokenizer.add_tokens('<sep>','<hl>')
print('* Max length of tokenizer: ', tokenizer.model_max_length)

print('* Loading Datasets...')
train_dataset = nlp.load_dataset('squad', split=nlp.Split.TRAIN)
#valid_dataset = nlp.load_dataset('squad', split=nlp.Split.VALIDATION)


print('* Preparing Datasets...')
hl = "<hl>"

# *Newly Update: split into train and valid data
print('\t Spliting data into train and valid...')
train_dataset_spliter = train_dataset.train_test_split(test_size=0.05)
print('\t The type of thee spliter: ', type(train_dataset_spliter))
train_data, valid_data = train_dataset_spliter['train'], train_dataset_spliter['test']
print('\t After spliting, the length of train dataset: ', len(train_data))
print('\t After spliting, the length of valid dataset: ', len(valid_data))

# extract useful inforamtionj
trainset = train_data.map(process_qg_text)
validset = valid_data.map(process_qg_text)
print('* The amount of train dataset:', len(trainset))
print('* The amount of valid dataset:', len(validset))


print('* Tokenizing data...')
# tokenize data
train_features = trainset.map(convert_to_features, batched=True)
valid_features = validset.map(convert_to_features, batched=True)

columns = ["source_ids", "target_ids", "attention_mask"]
train_features.set_format(type='torch', columns=columns)
valid_features.set_format(type='torch', columns=columns)

torch.save(train_features, 'data/train_v1.pt')
logger.info(f"saved train dataset at data/train_v1.pt")
print("saved train dataset at data/train_v1.pt")

torch.save(valid_features, 'data/valid_v1.pt')
logger.info(f"saved validation dataset at data/valid_v1.pt")
print("saved validation dataset at data/valid_v1.pt")


# Define Model and Train Model
model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model)
print('* Load model from: ', pretrained_model)
#print('\t Model structure: ', model)

batch_size = 16
targs = TrainingArguments(
    do_train=True,
    do_eval=True,
    local_rank=-1,
    evaluation_strategy=evaluation_strategy,
    eval_steps = eval_steps,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    num_train_epochs=num_epochs,
    fp16=False,
    output_dir = 'training_v1/',
    save_steps = save_steps
)

data_collator = T2TDataCollator(
    tokenizer=tokenizer,
    model_type='bart',
    mode='training',
)
print('* You are using model type: ', data_collator.model_type)

trainer = Trainer(
    model=model,
    args=targs,
    train_dataset=train_features,
    eval_dataset=valid_features,
    data_collator=data_collator,
    label_smoothing=0
)

if targs.do_train:
    trainer.train()
    trainer.save_model()


