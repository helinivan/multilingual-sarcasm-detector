# Multilingual Sarcasm Detector

This is a multilingual sarcasm detection model trained on news article titles from news papers in English, Dutch and Italian.

The models in this repository demonstrate that we can achieve state-of-the-art performance in sarcasm detection task by utilizing BERT and news article titles. BERT model utilizing English news articles achieved a F1-score of **92%**, whereas a mBERT model utilizing all news articles from English, Dutch and Italian achieved a F1-score of **87%**
 
The dataset generated utilizes both news articles from actual news sources, and news articles from sarcastic/satirical news sources. News article titles are either gathered from publicly available datasets or scraped directly from the newspaper websites. 
 
The multilingual sarcasm detection dataset is publicly available on [HuggingFace](https://huggingface.co/datasets/helinivan/sarcasm_headlines_multilingual).


The repository contains trainers for:
```
1. Monolingual BERT and SVM classifier models in English, Dutch and Italian
2. Multilingual BERT and SVM classifier models for English + Dutch + Italian
3. Multilingual subset BERT models where one or two languages are used in training,
   and the remaining languages are utilized in testing
```

Pre-trained BERT models used:
1. [bert-base-uncased](https://huggingface.co/bert-base-multilingual-uncased) (monolingual-bert-en)
2. [dbmdz/bert-base-italian-uncased](https://huggingface.co/dbmdz/bert-base-italian-uncased) (monolingual-bert-it)
3. [GroNLP/bert-base-dutch-cased](https://huggingface.co/GroNLP/bert-base-dutch-cased) (monolingual-bert-nl)
4. [bert-base-multilingual-uncased](https://huggingface.co/bert-base-multilingual-uncased) (multilingual models)

## Evaluation

Performance of the monolingual sarcasm detection models:

Model                                   | F1 
---------------------------------------- | :-------------: 
monolingual-bert-en |  **92.38** 
monolingual-bert-it | 88.26 
monolingual-bert-nl | 83.02 
monolingual-svm-en | 81.04 
monolingual-svm-it | 76.36 
monolingual-svm-nl |  75.52 

Performance of the multilingual sarcasm detection models:

Model                                   | F1 
-------------------------------------| :-----: 
multilingual-bert-all |  **87.23** 
multilingual-svm-all | 77.79 


In particular, the results of the research highlight the capability of both monolingual as well as multilingual models in sarcasm detection task. Moreover, from the findings we can observe that in multilingual setting, the models perform well on languages that have also been including in training set but the mBERT struggles with cross-lingual transfer in sarcasm detection task, and therefore there is a significant drop in performance when utilizing a subset multilingual models on unseen languages that have been excluded from training.


Performance of the subset sarcasm detection models based in bert-base-multilingual-uncased:

Model                                  | Languages trained | Languages tested |  F1 trained | F1 tested 
---------------------------------------- | :-------------: | :----------------: | :----------------: | :----------------:
multilingual-it | it | nl, en | 90.23 | **70.17**
multilingual-nl | nl | it, nl | 86.54 | 65.54
multilingual-en | en | nl, it |  90.68 | 58.67
multilingual-nl-it | nl, it | en | 88.02 | **71.99**
multilingual-en-nl | en, nl | it | 90.47 | 70.08
multilingual-en-it | en, it | nl | 89.96 | 65.90


In folder `results`, you can find evaluation metrics on evaluation set and test set, as well as confusion matrices and best hyperparameters.


## Setup

The `requirements.txt` file includes all necessary dependencies to run the code of this repository.

```
pip install -r requirements.txt
```

## Train

Trainer modules can be found in `trainers` folder. From there, execute the following commands, depending which model types should be trained.

```
python bert_trainer.py
```

```
python svm_trainer.py
```

## Predict

Following models are hosted on HuggingFace and publicly available:

1. [monolingual-bert-en](https://huggingface.co/helinivan/english-sarcasm-detector)
2. [monolingual-bert-it](https://huggingface.co/helinivan/italian-sarcasm-detector)
3. [monolingual-bert-nl](https://huggingface.co/helinivan/dutch-sarcasm-detector)
4. [multilingual-bert-all](https://huggingface.co/helinivan/multilingual-sarcasm-detector)

In file `predict.py`, an example is provided to show how to access and load the models from HuggingFace to local machine.
