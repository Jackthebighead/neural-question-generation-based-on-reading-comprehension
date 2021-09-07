# Import Packages
import os
import nlp
import torch 
from torch import nn
import logging
import datasets
from dataclasses import dataclass, field
from typing import Any, Union, Dict, List, Optional
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import Trainer as HFTrainer
from transformers import TrainingArguments
from transformers import HfArgumentParser

# Hyperparameters
device = 'cuda' if torch.cuda.is_available else 'cpu'
print('* You are using device: ', device)
logger = logging.getLogger(__name__)

model_checkpoint = "facebook/bart-base"  # "facebook/bart-base"
model_test_checkpoint = 'training_v1/'

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
print('* Your hyperparameter settings:\n')
print('\tPRETRAINED_MODEL:', model_checkpoint)
print('\tRESUME FROM: ', model_test_checkpoint)

# Define Functions
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
        ), "self.model.config.decoder_start_token_id has to be defined."

        # shift the inputs to the right 
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        assert torch.all(shifted_input_ids >= 0).item(), "Verify that `labels` has only positive values and -100"

        return shifted_input_ids


# Load Model
print('* Load Model...')
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
print('\t Model structure: ', model)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenizer.add_tokens('<sep>','<hl>')
print('\t Max length of tokenizer: ', tokenizer.model_max_length)
hl = "<hl>"
sep = "<sep>"

# Load Valid Dataset
print('* Loading Datasets...')
#train_dataset = nlp.load_dataset('squad', split=nlp.Split.TRAIN)
valid_dataset = nlp.load_dataset('squad', split=nlp.Split.VALIDATION)

print('* Preparing Datasets...')
hl = "<hl>"

# extract useful inforamtionj
#trainset = train_data.map(process_qg_text)
validset = valid_dataset.map(process_qg_text)
#print('* The amount of train dataset:', len(trainset))
print('* The amount of valid dataset:', len(validset))


print('* Tokenizing data...')
# tokenize data
#train_features = trainset.map(convert_to_features, batched=True)
valid_features = validset.map(convert_to_features, batched=True)

columns = ["source_ids", "target_ids", "attention_mask"]
#train_features.set_format(type='torch', columns=columns)
valid_features.set_format(type='torch', columns=columns)

#torch.save(train_features, 'data/train_v1.pt')
#logger.info(f"saved train dataset at data/train_v1.pt")
#print("saved train dataset at data/train_v1.pt")

#torch.save(valid_features, 'data/valid_v1.pt')
#logger.info(f"saved validation dataset at data/valid_v1.pt")
#print("saved validation dataset at data/valid_v1.pt")


data_collator = T2TDataCollator(
    tokenizer=tokenizer,
    model_type='bart',
    mode="inference"
)
print('* You are using model type: ', data_collator.model_type)

# Inference Model: use valid dataset to inference
loader = torch.utils.data.DataLoader(valid_features, batch_size=32, collate_fn=data_collator)
model_1 = AutoModelForSeq2SeqLM.from_pretrained(model_test_checkpoint)
predictions = get_predictions(
    model=model_1,
    tokenizer=tokenizer,
    data_loader=loader,
    num_beams=4,
    max_length=32
)
with open('inference_v1/inference_2.txt', 'w') as f:
    f.write("\n".join(predictions))
    
with open('inference_v1/ground_truth_2.txt', 'w') as f:
    f.write("\n".join(valid_features['question']))
    
# with open('inference/answers.txt', 'w') as f:
#     f.write("\n".join(valid_features['answers']['text']))

# Inference
# !pip install git+https://github.com/Maluuba/nlg-eval.git@master
from nlgeval import compute_metrics, compute_individual_metrics
print('testing on checkpoint:', model_test_checkpoint)
metrics_dict = compute_metrics(hypothesis='inference_v1/inference_2.txt', references=['inference_v1/ground_truth_2.txt'])
print('* Process Finished!')