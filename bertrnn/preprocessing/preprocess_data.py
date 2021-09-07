import json
import pickle
import nlp

from transformers import BertTokenizer


class Preprocess:

    def __init__(self, squad_type, bert_model, train_or_valid='train'):
        
        self.bert_model = bert_model
        self.squad_type = squad_type
        self.train_or_valid = train_or_valid
        self.hl = '<hl>'
        self.sep = "<sep>"
        
        if self.train_or_valid == 'train':
            print('\t Loading dataset: ', self.squad_type)
            dataset = nlp.load_dataset(self.squad_type, split=nlp.Split.TRAIN)
            dataset = _filter_empty_examples(dataset)
            print('\t The amount of training data: ', len(dataset))
        else:
            print('\t Loading dataset: ', self.squad_type)
            dataset = nlp.load_dataset(self.squad_type, split=nlp.Split.VALIDATION)
            dataset = _filter_empty_examples(dataset)
            print('\t The amount of valid data: ', len(dataset))
        
        print('* Preprocessing data...')
        input, output = _extract_squad_data(dataset, self.hl)
        print('* Tokenizing data...')
        self.data = _tokenize_data(input, output, self.bert_model)
        

    def save(self, path):
        print('* Saving processed data...')
        with open(path, 'wb') as write_file:
            pickle.dump(self.data, write_file)



def _filter_empty_examples(examples):
    '''
    Designed for SQuAD2.0 dataset since the unanswerable questions could not be examples in qg.
    '''
    return examples.filter(lambda example: len(example['answers']['text'])>0)


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


def _extract_squad_data(data, hl):
    
    # perform: data = dataset.map(process_qg_text)
    input, output = [], []
    for example in data:
        context = example['context']
        question = example['question']
        answer = example['answers']
        answer_text = answer['text'][0]
        start_pos, end_pos = get_correct_alignement(context, answer)
        qg_input = f"{context[:start_pos]} {hl} {answer_text} {hl} {context[end_pos:]}"
        qg_target = f"{question}"
        qg_input = qg_input + " </s>"
        qg_target = qg_target + " </s>"
        
        input.append(qg_input)
        output.append(qg_target)
    return input, output


# where problems may happens
def _tokenize_data(input, output, bert_model):
    tokenizer = BertTokenizer.from_pretrained(bert_model)
    # add special tokens
    tokenizer.add_tokens('<sep>','<hl>')
    
    data = tokenizer.batch_encode_plus(input, padding=True, return_tensors='pt') 
    out_dict = tokenizer.batch_encode_plus(output, padding=True, return_tensors='pt')

    data['output_ids'] = out_dict['input_ids']
    data['output_len'] = out_dict['attention_mask'].sum(dim=1)
    data['input_len'] = data['attention_mask'].sum(dim=1)

    idx = (data['input_len'] <= 512)
    #print(idx)
    in_m = max(data['input_len'][idx])
    out_m = max(data['output_len'][idx])

    data['input_ids'] = data['input_ids'][idx, :in_m]
    data['attention_mask'] = data['attention_mask'][idx, :in_m]
    data['token_type_ids'] = data['token_type_ids'][idx, :in_m]
    data['input_len'] = data['input_len'][idx]

    data['output_ids'] = data['output_ids'][idx, :out_m]
    data['output_len'] = data['output_len'][idx]

    return data



if __name__ == '__main__':
    print('* Prepare training data...')
    dataset = Preprocess('squad_v2', 'bert-base-cased', 'train')
    dataset.save(f'../dataset_squad/squad-v2-train.json')
    print('* Training data saved')
    
    print('* Prepare valid data...')
    dataset = Preprocess('squad_v2', 'bert-base-cased', 'valid')
    dataset.save(f'../dataset_squad/squad-v2-valid.json')
    print('* Valid data saved')
