## Introduction 
- We introduced a CoLA demo to demonstrate how to apply L1-norm weight sparser to BERT.

## Run

### DownLoad CoLA dataset
- CoLA: The Corpus of Linguistic Acceptability consists of English acceptability judgments drawn from books and journal articles on linguistic theory. Each example is a sequence of words annotated with whether it is a grammatical English sentence. More details can be get from [paper](https://arxiv.org/abs/1805.12471)
- how to get?
  - `python download.py`
  - `unzip ./cola_public_1.1.zip`

### Install Requirements
- `pip install -r requirements.txt`

### fine-tuning
- `python main.py sconfig.yaml`

## Results
- sratio = #zeros / #totol\_params
- Only BERT-Encoder be sparsed, excludes embedding & heads
- For convenience, we use accuracy as the metric here

model | sparser | sratio=0.25 | sratio=0.5 | sratio=0.75 | sratio=1.0 |
--- | --- | --- |  --- | --- | --- |
bert-base-uncased | l1-norm | 0.84 | 0.82 | 0.70 | 0.70 |
