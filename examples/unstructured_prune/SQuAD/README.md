## Introduction 
- We introduced a SQuAD demo to demonstrate how to apply L1-norm weight sparser to BERT.

## Run

### Install Requirements
- `pip install -r requirements.txt`

### fine-tuning
- `python main.py sconfig.yaml`

## Results
- sratio = #zeros / #totol\_params
- Only BERT-Encoder be sparsed, excludes embedding & heads
- For convenience, we use f1-score as the metric here

model | sparser | sratio=0.0 | sratio=0.25 | sratio=0.5 | sratio=0.75 | sratio=1.0 |
--- | --- | --- |  --- | --- | --- | --- |
bert-base-uncased | l1-norm |  88.57 | 88.09 | 86.98 | 75.42 | 10.46 |
