## Introduction 
- We introduced a CoLA demo to demonstrate how to apply PTQ to BERT.

## Run

### Install Requirements
- `pip install -r requirements.txt`

### Training
- run `python main.py finetuning` to get a checkpoint

### Post Training Quantization
- `python main.py postquant qconfig.yaml ./checkpoint.pth.tar`

## Results
- we use f1-score as the metric here

model | float | 8w8f |
--- | --- | --- |
bert-base-uncased | 88.22 | 87.47 |
