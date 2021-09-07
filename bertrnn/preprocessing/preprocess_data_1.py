import json
import pickle
import nlp

from transformers import BertTokenizer


class Preprocess:

    def __init__(self, squad_type, bert_model):
        
        self.bert_model = bert_model
        self.squad_type = squad_type
        self.hl = '<hl>'
        self.sep = "<sep>"
        
        print('\t Loading train data from SQuAD version: ', self.squad_type)
        dataset = nlp.load_dataset(self.squad_type, split=nlp.Split.TRAIN)
        dataset = _filter_empty_examples(dataset)
        print('\t The amount of data before splitting: ', len(dataset))
        train_data, valid_data = _split_train_valid_data(dataset)
        
        print('* Preprocessing train data...')
        train_input, train_output = _extract_squad_data(train_data, self.hl)
        print('* Tokenizing train data...')
        self.train_data = _tokenize_data(train_input, train_output, self.bert_model)

        print('* Preprocessing valid data...')
        valid_input, valid_output = _extract_squad_data(valid_data, self.hl)
        print('* Tokenizing valid data...')
        self.valid_data = _tokenize_data(valid_input, valid_output, self.bert_model)

    def save(self, path_train, path_valid):
        print('* Saving processed train data...')
        with open(path_train, 'wb') as write_file_train:
            pickle.dump(self.train_data, write_file_train)
        print('* Saving processed valid data...')
        with open(path_valid, 'wb') as write_file_valid:
            pickle.dump(self.valid_data, write_file_valid)


def _split_train_valid_data(train_dataset_clean):
    print('\t Spliting data into train and valid...')
    train_dataset_spliter = train_dataset_clean.train_test_split(test_size=0.05)
    print('\t The type of thee spliter: ', type(train_dataset_spliter))
    train_data, valid_data = train_dataset_spliter['train'], train_dataset_spliter['test']
    print('\t After spliting, the length of train dataset: ', len(train_data))
    print('\t After spliting, the length of valid dataset: ', len(valid_data))
    
    return train_data, valid_data
    

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

    data = tokenizer.batch_encode_plus(input, padding=True, truncation=True, return_tensors='pt') 
    out_dict = tokenizer.batch_encode_plus(output, padding=True, truncation=True, return_tensors='pt')

    print('\t Tokenizer max length:', len(data['input_ids'][0]))
    
    data['output_ids'] = out_dict['input_ids']
    data['output_len'] = out_dict['attention_mask'].sum(dim=1)
    data['input_len'] = data['attention_mask'].sum(dim=1)

    idx = (data['input_len'] <= 1024)
    #print(idx)
    in_m = max(data['input_len'][idx])
    out_m = max(data['output_len'][idx])
    
    print('\t Total num of data:', len(idx))
    print('\t Max length of input and output sentence:', in_m, out_m)

    data['input_ids'] = data['input_ids'][idx, :in_m]
    data['attention_mask'] = data['attention_mask'][idx, :in_m]
    data['token_type_ids'] = data['token_type_ids'][idx, :in_m]
    data['input_len'] = data['input_len'][idx]

    data['output_ids'] = data['output_ids'][idx, :out_m]
    data['output_len'] = data['output_len'][idx]

    return data



if __name__ == '__main__':
    print('* Prepare train and valid data...')
    dataset = Preprocess('squad_v2', 'bert-base-cased')
    dataset.save(f'../dataset_squad/train_data.json', f'../dataset_squad/valid_data.json')
    print('* Train and Valid data saved')
