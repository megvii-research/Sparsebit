import os
import argparse
import torch
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from datasets import load_dataset
import transformers
from transformers.optimization import AdamW
from transformers import LlamaTokenizer, get_linear_schedule_with_warmup
from transformers.trainer_utils import seed_worker, set_seed
from transformers.trainer_pt_utils import IterableDatasetShard
from qlora import get_peft_qmodel
from utils import load_qllama
from tqdm import tqdm
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
)


def main(args):

    BATCH_SIZE = args.batch_size
    GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // args.micro_batch_size
    EPOCHS = 3  # we don't need 3 tbh
    LEARNING_RATE = 3e-4  # the Karpathy constant
    CUTOFF_LEN = 256  # 256 accounts for about 96% of the data
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    TRAIN_SET_SIZE = None
    VAL_SET_SIZE = 2000

    # Need to initialize RPC framework first.
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    torch.distributed.rpc.init_rpc("worker", rank=0, world_size=1)

    config = transformers.AutoConfig.from_pretrained(args.model)

    pp_kwargs = {
        "chunks": args.chunks,
        "checkpoint": args.pp_checkpoint,
    }

    model = load_qllama(config, args.cachedir, pp_kwargs)
    model = prepare_model_for_int8_training(model)
    model.seq_len = 2048
    peft_func = get_peft_qmodel
    tokenizer = LlamaTokenizer.from_pretrained(args.model, add_eos_token=True)
    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="QUANT_CAUSAL_LM",
    )
    model = peft_func(model, config)
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    data = load_dataset("json", data_files="alpaca_data.json")

    train_val = data["train"].train_test_split(
        train_size=TRAIN_SET_SIZE, test_size=VAL_SET_SIZE, shuffle=False, seed=42
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

    train_data = train_data.map(lambda x: tokenize(generate_prompt(x)))
    val_data = val_data.map(lambda x: tokenize(generate_prompt(x)))

    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)

    signature_columns = ["input_ids", "attention_mask"]
    ignored_columns = list(set(train_data.column_names) - set(signature_columns))

    train_dataset = train_data.remove_columns(ignored_columns)

    generator = torch.Generator()
    generator.manual_seed(42)
    set_seed(42)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.micro_batch_size,
        sampler=RandomSampler(train_dataset, generator=generator),
        collate_fn=data_collator,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )

    if isinstance(train_dataloader, DataLoader) and isinstance(
        train_dataloader.sampler, DistributedSampler
    ):
        train_dataloader.sampler.set_epoch(0)
    elif hasattr(train_dataloader, "dataset") and isinstance(
        train_dataloader.dataset, IterableDatasetShard
    ):
        train_dataloader.dataset.set_epoch(0)

    epoch_iterator = train_dataloader

    model.print_trainable_parameters()

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=0.0,
        eps=1e-8,
    )
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=EPOCHS*len(epoch_iterator) // GRADIENT_ACCUMULATION_STEPS,
    )

    scaler = torch.cuda.amp.GradScaler()
    loss_fct = torch.nn.CrossEntropyLoss()

    # IMPORTANT! model.eval() -> model.train() enable requant 4-bit weights
    model.eval()
    model.train()

    for epoch in range(EPOCHS):
        step = 0
        accumulated_loss = 0
        for inputs in tqdm(epoch_iterator):
            step += 1
            with torch.cuda.amp.autocast(cache_enabled=True, dtype=torch.float16):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                outputs["logits"] = outputs["logits"].float()
                labels = labels.to(outputs["logits"].device)
                shift_logits = outputs["logits"][..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                shift_logits = shift_logits.view(-1, shift_logits.shape[-1])
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)

            loss /= GRADIENT_ACCUMULATION_STEPS
            accumulated_loss += loss.item()
            scaler.scale(loss).backward()

            if step % GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0
                )
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                model.zero_grad()
                tqdm.write(
                    "{"
                    + "'loss': {0:1.4f}, 'learning_rate': {1:2.6f}, 'epoch': {2:3.2f}".format(
                        accumulated_loss,
                        optimizer.param_groups[0]["lr"],
                        step / len(epoch_iterator)+epoch,
                    )
                    + "}"
                )
                accumulated_loss = 0

    print("Peak memory usage for GPUs: ", end="")
    for i in range(len(model.model.devices)):
        print(
            "cuda:{}: {}, ".format(
                i, sizeof_fmt(torch.cuda.memory_stats(i)["allocated_bytes.all.peak"])
            ),
            end="",
        )
    print()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="model name")
    parser.add_argument("cachedir", type=str, help="path to 4bit checkpoint")
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training."
    )
    parser.add_argument(
        "--micro_batch_size",
        type=int,
        default=64,
        help="Micro batch size for gradient accumulating.",
    )
    parser.add_argument(
        "--chunks", type=int, default=4, help="Number of trunks for pp."
    )
    parser.add_argument(
        "--pp_checkpoint", type=str, default="except_last", help="checkpoint for pp`."
    )
    args = parser.parse_args()
    main(args)
