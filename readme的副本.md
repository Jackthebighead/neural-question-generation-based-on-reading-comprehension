## Readme File of MSBD5014A Independent Project

`Author: YUAN Yanzhe, MSc Big Data Technology`

`Student No: 20728555`

`Email: yyuanar@connect.ust.hk`



### Introduction of the Codes

The codes contains two models implemented in this project.

- bart

  - Training codes of bart on SQuAD1.0 and 2.0
  - Run the code on GPU mode
  - Train the model: **run_bart_squadv2.py** (run_bart_squadv1.py)
    - by running the file, the SQuAD data will be downloaded from HuggingFace, and then saved at 'data/'
    - the training model will be saved at 'training_v2/'
  - Test the model: **test_bart_squadv2.py** (test_bart_squadv1.py)
    - During the testing, the model prediction as well as the ground truth will be saved at 'inference_v2/', make sure to makedir inference_v2 first.
  - Others:
    - **sent_emb.py**:
      - get sentence embedding of the ground truth and the inference.
      - compare the cosine similarity between them
    - **nlg_eval.py**: a demo code for using nlg_eval package.

- bertrnn

  - Training codes of bertrnn on SQuAD

  - Run the code on GPU mode

  - Data Prepocessing:

    - cd 'preprocessing/'
    - run **preprocess_data.py**
    - the results can be seen in 'dataset_squad/'

  - Train the model: **run_main.py**

  - Modify the hyperparameters: **config.py**

    

