### Introduction
- [alpaca-lora](https://github.com/tloen/alpaca-lora) is a great project which allows to run instuct-tuning on a single RTX4090 within hours. After instruct-tuning, an instruct model of similar quality to text-davinci-003 that can be obtained.
- However, the larger foundation model, the better the instruction results can be obtained. And we hope that everyone can enjoy this benefit. Therefore, we provide alpaca-qlora, which quantizes the backbone into 4bit while keep lora-parameters as fp16. 
- In alpaca-qlora, the GPU memory of about half model size will be released(for example, llama-7B will releases 3.5GB). When computing resources are insufficient, it can alleviate the demand; even in the case of sufficient computing resources, alpaca-qlora can help to expand the CUTOFF\_LEN which maybe improve your instuct-tuning results or increase macro-batch-size to reduce your training time.

### Install 
- 1. Install dependencies
`pip install -r requirements.txt`

- 2. If bitsandbytes doesn't work, [install it from source.](https://github.com/TimDettmers/bitsandbytes/blob/main/compile_from_source.md) Windows users can follow [these instructions](https://github.com/tloen/alpaca-lora/issues/17).

- 3. Install CUDA cutlass
  - `git clone https://github.com/NVIDIA/cutlass`
  - `cd /path/to/repo/cuda/`
  - `ln -s /path/to/cutlass/ .`
  - `./build_cutlass.sh`
  - `./environment.sh`

### Usage
#### how to quant backbone?
- go to [qllama](https://github.com/megvii-research/Sparsebit/tree/main/large_language_models/llama/quantization) to get quant backbone
- you can also download a checkpoint [llama-7B](https://drive.google.com/file/d/1qcwDiHbJAZNd4l2SFtshrEs2G1VHr6MK/view?usp=share_link) as the quant backbone(pack32).
- convert the weight dtype of quant backbone from torch.int32 to torch.int8: `python3 convert_pack32topack8.py /path/to/quant-backbone-pack32 /path/to/output-quant-backbone-pack8`

#### Training on single 2080ti
- `python3 finetune.py`

#### Training on 8gpu 2080ti with Pipeline Parallelism(PP)
- LLaMA-7b: `python3 finetune_pp.py decapoda-research/llama-7b-hf /path/to/llama7b-pack8 --chunks 4 --pp_checkpoint except_last --micro_batch_size 32`
- LLaMA-65b: `python3 finetune_pp.py decapoda-research/llama-65b-hf /path/to/llama65b-pack8 --chunks 8 --pp_checkpoint except_last --micro_batch_size 16`

#### Inference
- `python3 generate.py`

### training on single 2080ti
- the data of gpu-memory from nvidia-smi

method | gpu-memory | micro-batch-size | gpu-hours
--- | --- | --- | --- |
alpaca-lora | 8.71G | 4 | 14.25h |
alpaca-qlora(ours) | 5.63G | 4 | 16h | 
alpaca-qlora(ours) | 8.09G | 16 | 11.5h | 

### Time cost per batch for training on 8gpu 2080ti with Pipeline Parallelism(PP)

|LLaMA-7b chunks|1|2|4|8|16|32|
|---|---|---|---|---|---|---|
|micro_batch=8|3.0h|2.2h|2.3h|4.3h|-|-|
|micro_batch=16|2.6h|1.8h|1.6h|2.3h|4.5h|-|
|micro_batch=32|OOM|1.6h|1.25h|1.45h|2.4h|4.5h|

|LLaMA-65b chunks|2|4|8|16|
|---|---|---|---|---|
|micro_batch=8|OOM|23h|27h|-|
|micro_batch=16|OOM|OOM|16h|23h|



### Results
#### sample-1
- Instruct: `Tell me about alpaca.`
- alpaca-qlora output: `Alpacas are members of the camelid family and are native to the Andes Mountains of South America. They are prized for their soft, luxurious fleece, which is used to make clothing, blankets, and other items. Alpacas are herbivores and graze on grasses, shrubs, and other vegetation. Alpacas are social animals and live in herds of up to 20 individuals. They are known for their docile nature and gentle temperament.`
- alpaca-lora output: `Alpacas are members of the camelid family and are native to the Andes Mountains of South America. They are known for their soft, luxurious fleece, which is used to make clothing, blankets, and other items. Alpacas are herbivores and graze on grasses and other plants. They are social animals and live in herds of up to 20 individuals.`

#### sample-2
- Instruct: List all Canadian provinces in alphabetical order
- alpaca-qlora output: `Alberta, British Columbia, Manitoba, New Brunswick, Newfoundland and Labrador, Northwest Territories, Nova Scotia, Ontario, Prince Edward Island, Saskatchewan, Yukon.`
- alpaca lora output: `Alberta, British Columbia, Manitoba, New Brunswick, Newfoundland and Labrador, Nova Scotia, Ontario, Prince Edward Island, Quebec, Saskatchewan.`

#### sample-3
- Instuct: Write a Python program that prints the first 10 Fibonacci numbers.
- alpaca-qlora output:
```
"""
This program prints the first 10 Fibonacci numbers.
"""

def fibonacci(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)

print(fibonacci(10))
```

- alpaca-lora output:
```
def fibonacci(n):
if n == 0:
    return 0
elif n == 1:
    return 1
else:
    return fibonacci(n-1) + fibonacci(n-2)
print(fibonacci(10))
```

### Acknowledgement
- We are grateful for these excellent projects and list them as follows:
  - [GPTQ](https://github.com/IST-DASLab/gptq)
  - [alpaca-lora](https://github.com/tloen/alpaca-lora)
