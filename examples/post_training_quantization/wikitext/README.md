## Introduction 
- We introduced a WikiText demo to demonstrate how to apply PTQ to GPT2.
- Notice that ACT2FN(modified LayerNorm) in GPT2MLP of official GPT2 model is replaced by nn.LayerNorm, which lead to slightly performance degragation (zero-shot ppl from 29.88 to 29.96)

## Run

### Install Requirements
- `pip install -r requirements.txt`

### Training(Optional)
- run `python main.py finetuning` to get a fine-tuned float checkpoint

### Post Training Quantization
- Zero-shot PTQ: `python main.py postquant qconfig.yaml`
- Fine-tuned PTQ: `python main.py postquant qconfig.yaml --checkpoint ./checkpoint.pth.tar`

## Results
- For convenience, we use perplexity as the metric here

model | float | 8w8f 
--- | --- | --- |
official gpt2-small|29.88|-|
zero-shot gpt2-small | 29.96 | 38.29 |
fine-tuned gpt2-small |  |  |
