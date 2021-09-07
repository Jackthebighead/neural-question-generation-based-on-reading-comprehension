# Neural Question Generation based on Reading Comprehension

This repository contains implementations of the following Seq2Seq models for Nueral Question Generation (NQG) task on SQuAD (both 1.0 and 2.0) datasets. 

The models are as following,

- **BERT+RNN(GRU/LSTM)**: with optimizations on attention, teacher forcing, beam search and copy mechanism(copy mechanism version tbc).
- **BART**: a Pre-trained Transformer based Seq2Seq model is fine-tuned.
- **T5**: a Pre-trained Transformer based Seq2Seq model is fine-tuned.



### Introduction

Question Generation (QG), which focuses on generating corresponding question based on given contexts from various inputs, is a subtask of Natural Language Generation. With the recent developments in Deep Learning and Pre-trained mod- els, Neural Qeustion Generation (NQG) attracts more and more research interests nowadays. Seq2Seq architecture, which allows the Encoder to comprehend the inputs and the Decoder to generate the outputs, can better utilize the context in- formation and thus generate better questions.

Seq2Seq based models are implemented for Question Generation on SQuAD dataset. BERT+RNN based model is trained, and a pre-trained BART model is fine-tuned for this QG task. As a result, BART performs better and is adopted as the final model for this task. Both automatic metrics and human judgements are acquired for evaluating the model performances as well as result analysis.



### Models and Evaluations

BERT+RNN based model is inspired by Machine Translation, where BERT is used as Encoder and RNN(LSTM/GRU) is used as Decoder. Optimizations like training strategy, attention type(additive, scaled dot-product, multivariate), teacher forcing ratio,  beam size are experimented or tuned. Furthermore, the idea of PGN, which uses the weighted sum of both copy and generative distributions to generate words, can be used as well. 

 [BART](https://arxiv.org/pdf/1910.13461.pdf) is a Transformer based Seq2Seq model, which can be decomposed into a bidirectional Encoder on noised input and an auto-regressive Decoder. The main contribution of BART is to use arbitrary noise to destroy the original text, and then the model is required to learn to reconstruct the original text. As the original paper said, BART can be regarded as the generalization of BERT and GPT-2. 

[T5](https://arxiv.org/pdf/1910.10683.pdf) is a Transformer based text-to-text framework, which achieve state-of-the-art results on many benchmarks covering summarization, question answering, text classification, and more.

Evaluations are done from two aspects: human judgements like human-defined principles and automatic metrics like BLEU, ROUGE, and METEOR, which can be used in [nlg-eval](https://github.com/Maluuba/nlg-eval).

One potential feasible is proposed. Sentence embedding similarity can be used as a embedding based metric to measure the performances of the generation results. And, sentence embedding can be obtained by models or structures like [Sentence-BERT](https://arxiv.org/pdf/1908.10084.pdf) (a siamese BERT structure) (can be used in a fine-tuned fashion on the corpus or feature based fashion). 



### Future Work

- Involving both automatic metrics and human judgements in the design of loss function.



### Notes on Reproduction

- Run the codes on GPU mode
- BART/T5 based
  - Train the model: **run_bart_squadv2.py** (run_bart_squadv1.py)
  - Test the model: **test_bart_squadv2.py** (test_bart_squadv1.py)
  - **sent_emb.py**: get sentence embedding of the ground truth and the inference and compare the cosine similarity between them.
  - **nlg_eval.py**: a demo code of using nlg_eval package.
- BERT+RNN based
  - Data Prepocessing:
    - cd 'preprocessing/'
    - run **preprocess_data.py**
    - then the results can be seen in 'dataset_squad/'
  - Train the model: **run_main.py**


