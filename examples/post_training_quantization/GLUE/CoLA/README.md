## Introduction 
- We introduced a CoLA demo to demonstrate how to apply PTQ to BERT.

## Run

### DownLoad CoLA dataset
- CoLA: The Corpus of Linguistic Acceptability consists of English acceptability judgments drawn from books and journal articles on linguistic theory. Each example is a sequence of words annotated with whether it is a grammatical English sentence. More details can be get from [paper](https://arxiv.org/abs/1805.12471)
- how to get?
  - `python download.py`
  - `unzip ./cola_public_1.1.zip`

### Install Requirements
- `pip install -r requirements.txt`

### Training
- run `python main.py finetuning` to get a checkpoint

### Post Training Quantization
- `python main.py postquant qconfig.yaml ./checkpoint.pth.tar`

## Results
- For convenience, we use accuracy as the metric here

model | float | 8w8f |
--- | --- | --- |
bert-base-uncased | 0.84 | 0.83 |
