### Introduction
- [alpaca-lora](https://github.com/tloen/alpaca-lora) is a great project which allows to run instuct-tuning on a single RTX4090 within hours. After instruct-tuning, an instruct model of similar quality to text-davinci-003 that can be obtained.
- However, the larger foundation model, the better the instruction results can be obtained. And we hope that everyone can enjoy this benefit. Therefore, we provide alpaca-qlora, which quantizes the backbone into 4bit while keep lora-parameters as fp16. 
- In alpaca-qlora, the GPU memory of about half model size will be released(for example, llama-7B will releases 3.5GB). When computing resources are insufficient, it can alleviate the demand; even in the case of sufficient computing resources, alpaca-qlora can help to expand the CUTOFF\_LEN which maybe improve your instuct-tuning results or increase macro-batch-size to reduce your training time.

### Install 
- 1. Install dependencies
`pip install -r requirements.txt`

- 2. If bitsandbytes doesn't work, [install it from source.](https://github.com/TimDettmers/bitsandbytes/blob/main/compile_from_source.md) Windows users can follow [these instructions](https://github.com/tloen/alpaca-lora/issues/17).

- 3. Install CUDA cutlass
`cd cuda`
`./build_cutlass.sh`
`./environment.sh`

- 4. modify transformers modeling\_llama for loading checkpoint
`cd /path/to/site-packages/transformers/models/llama`
`vim modeling_llama.py, go to line98`
`replace self.register_buffer("inv_freq", inv_freq) with self.inv_freq = inv_freq`

### Usage
#### how to quant backbone?
- go to [qllama](https://github.com/megvii-research/Sparsebit/tree/main/large_language_models/llama/quantization) to get quant backbone
- pack quant backbone to torch.int8: `python3 convert.py /path/to/quant-backbone /path/to/output-quant-backbone`
- you can download a [llama-7B] checkpoint to run.

#### Training
- `python3 finetun.py`

#### Inference
- `python3 generate.py`


### training on single 2080ti
- the data of gpu-memory from nvidia-smi

method | gpu-memory | macro-batch-size | gpu-hours | final-loss
--- | --- | --- | --- | --- 
alpaca-lora | 8.71G | 4 | 14.25h |
alpaca-qlora(ours) | 5.63G | 4 | 16h | 
alpaca-qlora(ours) | 8.09G | 16 | 11.5h | 


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


