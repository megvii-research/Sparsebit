import os
import argparse
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


def main(args):

    # optimized for RTX 4090. for larger GPUs, increase some of these?
    MICRO_BATCH_SIZE = args.micro_batch_size
    BATCH_SIZE = args.batch_size
    GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
    EPOCHS = 3  # we don't need 3 tbh
    LEARNING_RATE = 3e-4  # the Karpathy constant
    CUTOFF_LEN = 256  # 256 accounts for about 96% of the data
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    TRAIN_SET_SIZE = None
    VAL_SET_SIZE = 2000

    if args.int4_backbone:
        config = transformers.AutoConfig.from_pretrained(
            args.model
        )
        model = load_qllama(
            config, args.int4_backbone
        )
        model.is_loaded_in_8bit = True  # hack for gradient-checkpoint
        model = prepare_model_for_int8_training(model)
        model.is_loaded_in_8bit = False
        model.seq_len = 2048
        peft_func = get_peft_qmodel
    else:
        model = LlamaForCausalLM.from_pretrained(
            args.model,
            load_in_8bit=True,
            device_map="auto",
        )
        model = prepare_model_for_int8_training(model)
        peft_func = get_peft_model

    tokenizer = LlamaTokenizer.from_pretrained(args.model, add_eos_token=True)

    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="QUANT_CAUSAL_LM" if args.int4_backbone else "CAUSAL_LM",
    )
    model = peft_func(model, config)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    data = load_dataset("yahma/alpaca-cleaned")

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
            logging_steps=10,
            logging_dir="runs/logs",
            logging_strategy="steps",
            optim="adamw_torch",
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

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="model name")
    parser.add_argument("--int4_backbone", type=str, default="", help="path to 4bit checkpoint, using int4 backbone if provided")
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training."
    )
    parser.add_argument(
        "--micro_batch_size", type=int, default=16, help="Batch size for training."
    )
    args = parser.parse_args()
    main(args)