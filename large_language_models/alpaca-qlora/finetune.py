import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset
import transformers
from utils import load_qllama, QuantLinear

assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)
from qlora import get_peft_qmodel


# optimized for RTX 4090. for larger GPUs, increase some of these?
DEBUG = False
QUANT = True
if DEBUG:
    MICRO_BATCH_SIZE = 2
    BATCH_SIZE = 2
else:
    MICRO_BATCH_SIZE = 16  # this could actually be 5 but i like powers of 2
    BATCH_SIZE = 128
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 3  # we don't need 3 tbh
LEARNING_RATE = 3e-4  # the Karpathy constant
CUTOFF_LEN = 256  # 256 accounts for about 96% of the data
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TRAIN_SET_SIZE = None
VAL_SET_SIZE = 2000
model_arch = "llama-7b"
model_cachedir = "./caches/{}/".format(model_arch)
if DEBUG:
    TRAIN_SET_SIZE = 2000
    VAL_SET_SIZE = 100

if QUANT:
    config = transformers.AutoConfig.from_pretrained(
        os.path.join(model_cachedir, "config.json")
    )
    model = load_qllama(
        config, os.path.join(model_cachedir, "{}_4w_pack8.pth.tar".format(model_arch))
    )
    model.is_loaded_in_8bit = True  # hack for gradient-checkpoint
    model = prepare_model_for_int8_training(model)
    model.is_loaded_in_8bit = False
    model.seq_len = 2048
    peft_func = get_peft_qmodel
else:
    model = LlamaForCausalLM.from_pretrained(
        "decapoda-research/llama-7b-hf",
        load_in_8bit=True,
        device_map="auto",
    )
    model = prepare_model_for_int8_training(model)
    peft_func = get_peft_model

# tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf", add_eos_token=True)
tokenizer = LlamaTokenizer.from_pretrained(
    os.path.join(model_cachedir, "tokenizer"), add_eos_token=True
)

config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="QUANT_CAUSAL_LM" if QUANT else "CAUSAL_LM",
)
model = peft_func(model, config)

tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
data = load_dataset("json", data_files="alpaca_data.json")

train_val = data["train"].train_test_split(
    train_size=TRAIN_SET_SIZE, test_size=VAL_SET_SIZE, shuffle=True, seed=42
)
train_data = train_val["train"]
val_data = train_val["test"]


def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
{data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Response:
{data_point["output"]}"""


def tokenize(prompt):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN + 1,
        padding="max_length",
    )
    return {
        "input_ids": result["input_ids"][:-1],
        "attention_mask": result["attention_mask"][:-1],
    }


train_data = train_data.shuffle().map(lambda x: tokenize(generate_prompt(x)))
val_data = val_data.shuffle().map(lambda x: tokenize(generate_prompt(x)))

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=100,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        # logging_steps=20,
        logging_steps=1,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=200,
        save_steps=200,
        output_dir="lora-alpaca",
        save_total_limit=3,
        load_best_model_at_end=True,
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False

old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
).__get__(model, type(model))

# IMPORTANT! model.eval() -> model.train() enable requant 4-bit weights
model.eval()
model.train()

trainer.train()
# res = trainer.evaluate()

model.save_pretrained("lora-alpaca")

print("\n If there's a warning about missing keys above, please disregard :)")


from functools import partial

# forward_values = {}
# def hook_fn_forward(module, inp, out, name=None):
#    forward_values[module] = {}
#    forward_values[module]["input"] = inp
#    forward_values[module]["output"] = out
#    print("debug forward")
#    from IPython import embed; embed()
#
# model.base_model.model.model.layers[-1].self_attn.q_proj.lora_A.register_forward_hook(partial(hook_fn_forward, name="q_lora_a"))
# model.base_model.model.model.layers[-1].self_attn.q_proj.lora_B.register_forward_hook(partial(hook_fn_forward, name="q_lora_b"))
# model.base_model.model.model.layers[-1].self_attn.v_proj.lora_A.register_forward_hook(partial(hook_fn_forward, name="v_lora_a"))
# model.base_model.model.model.layers[-1].self_attn.v_proj.lora_B.register_forward_hook(partial(hook_fn_forward, name="v_lora_b"))

# register backward hook for lora
backward_values = {}


def hook_fn_backward(module, inp_grad, out_grad, name=None):
    backward_values[module] = {}
    backward_values[module]["input"] = inp_grad
    backward_values[module]["output"] = out_grad
    print("debug backward")
    from IPython import embed

    embed()


# model.base_model.model.model.layers[-1].self_attn.q_proj.lora_A.register_backward_hook(partial(hook_fn_backward, name="q_lora_a"))
# model.base_model.model.model.layers[-1].self_attn.q_proj.lora_B.register_backward_hook(partial(hook_fn_backward, name="q_lora_b"))
# model.base_model.model.model.layers[-1].self_attn.v_proj.lora_A.register_backward_hook(partial(hook_fn_backward, name="v_lora_a"))
# model.base_model.model.model.layers[-1].self_attn.v_proj.lora_B.register_backward_hook(partial(hook_fn_backward, name="v_lora_b"))


## 防呆测试
"""
tokenizer = LlamaTokenizer.from_pretrained(
    "decapoda-research/llama-7b-hf", add_eos_token=True
)

def get_wikitext2():
    from datasets import load_dataset
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    tokenizer = LlamaTokenizer.from_pretrained("/data/llama/hf/7b/tokenizer", use_fast=False)
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
    return testenc

testloader = get_wikitext2()
DEV = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="QUANT_CAUSAL_LM" if QUANT  else "CAUSAL_LM"
)
model = peft_func(model, config)
model.seqlen = 2048

print("The Perplexity on wikiText2: ")
llama_eval(model, testloader, DEV)
"""
