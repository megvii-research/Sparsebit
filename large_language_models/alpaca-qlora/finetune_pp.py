import os
import argparse
import torch
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler
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
    get_peft_model_state_dict,
    LoraConfig,
)
import math
from tensorboardX import SummaryWriter


def main(args):

    BATCH_SIZE = args.batch_size
    GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // args.micro_batch_size
    EPOCHS = 3  # we don't need 3 tbh
    LEARNING_RATE = args.lr  # the Karpathy constant
    CUTOFF_LEN = 256  # 256 accounts for about 96% of the data
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    TRAIN_SET_SIZE = None
    VAL_SET_SIZE = 2000
    SAVE_PATH="runs/llama-{}-qlora-lr{:.1e}".format(args.model.split("/")[-1].split("-")[1], args.lr)
    logging_steps=10
    eval_steps=200
    save_steps=200

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
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference
    data = load_dataset("yahma/alpaca-cleaned")

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
    val_dataset = train_data.remove_columns(ignored_columns)

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
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.micro_batch_size,
        sampler=SequentialSampler(val_dataset),
        collate_fn=data_collator,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )

    if hasattr(train_dataloader, "dataset") and isinstance(
        train_dataloader.dataset, IterableDatasetShard
    ):
        train_dataloader.dataset.set_epoch(0)

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
        num_training_steps=EPOCHS * len(train_dataloader) // GRADIENT_ACCUMULATION_STEPS,
    )

    scaler = torch.cuda.amp.GradScaler()
    loss_fct = torch.nn.CrossEntropyLoss()
    writer = SummaryWriter(SAVE_PATH)

    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

    # IMPORTANT! model.eval() -> model.train() enable requant 4-bit weights
    model.eval()
    model.train()

    for epoch in range(EPOCHS):
        step = 0
        accumulated_loss = 0
        model.train()
        for inputs in tqdm(train_dataloader):
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
                if (step//GRADIENT_ACCUMULATION_STEPS + len(train_dataloader)//GRADIENT_ACCUMULATION_STEPS * epoch)%logging_steps == 0:
                    tqdm.write(
                        "{"
                        + "'loss': {0:1.4f}, 'learning_rate': {1:2.6f}, 'epoch': {2:3.2f}".format(
                            accumulated_loss/logging_steps,
                            optimizer.param_groups[0]["lr"],
                            step / len(train_dataloader) + epoch,
                        )
                        + "}"
                    )
                    writer.add_scalar('train/learning_rate', optimizer.param_groups[0]["lr"], step//GRADIENT_ACCUMULATION_STEPS + len(train_dataloader)//GRADIENT_ACCUMULATION_STEPS * epoch)
                    writer.add_scalar('train/loss', accumulated_loss/logging_steps, step//GRADIENT_ACCUMULATION_STEPS + len(train_dataloader)//GRADIENT_ACCUMULATION_STEPS * epoch)
                    accumulated_loss = 0
                if (step//GRADIENT_ACCUMULATION_STEPS + len(train_dataloader)//GRADIENT_ACCUMULATION_STEPS * epoch)%eval_steps == 0:
                    accumulated_loss = 0
                    model.eval()
                    s=0
                    for inputs in tqdm(val_dataloader):
                        s+=1
                        with torch.no_grad():
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

                            accumulated_loss += loss.item()


                    print('Eval_loss:', str(accumulated_loss/s))
                    writer.add_scalar('eval/loss', accumulated_loss/s, step//GRADIENT_ACCUMULATION_STEPS + len(train_dataloader)//GRADIENT_ACCUMULATION_STEPS * epoch )

    model.save_pretrained(SAVE_PATH)

    print("\n If there's a warning about missing keys above, please disregard :)")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="model name")
    parser.add_argument("cachedir", type=str, help="path to 4bit checkpoint")
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training."
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4, help="Max learning rate for training."
    )
    parser.add_argument(
        "--micro_batch_size",
        type=int,
        default=32,
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
