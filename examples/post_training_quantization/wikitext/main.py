import random
import math
import numpy as np
import time
import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
from sparsebit.quantization import QuantModel, parse_qconfig

from model import ModifiedGPT2LMHeadModel

def build_dataset(block_size=1024, calibration_size=None, batch_size=32):
    datasets = load_dataset("wikitext", "wikitext-103-v1")

    print("Loading GPT2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    def tokenize_function(examples):
        return tokenizer(examples["text"])

    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        num_proc=None,
        remove_columns=["text"],
        load_from_cache_file=True,
    )

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=None,
        load_from_cache_file=True,
    )

    print("start train dataset loading!")
    start_time = time.time()
    lm_dataset_train = lm_datasets["train"]
    input_ids = []
    attention_masks = []
    labels = []
    if calibration_size:
        idxs = random.sample(range(len(lm_dataset_train)), calibration_size)
    else:
        idxs = range(len(lm_dataset_train))

    for i in tqdm(idxs):
        input_ids.append(torch.tensor(lm_dataset_train[i]["input_ids"]).unsqueeze(0))
        attention_masks.append(torch.tensor(lm_dataset_train[i]["attention_mask"]).unsqueeze(0))
        labels.append(torch.tensor(lm_dataset_train[i]["labels"]).unsqueeze(0))
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.cat(labels, dim=0)
    train_dataset = TensorDataset(input_ids, attention_masks, labels)
    end_time = time.time()
    print("Finish loading train dataset, time cost:", str(end_time-start_time))

    print("start val dataset loading!")
    start_time = time.time()
    lm_dataset_val = lm_datasets["validation"]
    input_ids = []
    attention_masks = []
    labels = []
    for i in tqdm(range(len(lm_dataset_val))):
        input_ids.append(torch.tensor(lm_dataset_val[i]["input_ids"]).unsqueeze(0))
        attention_masks.append(torch.tensor(lm_dataset_val[i]["attention_mask"]).unsqueeze(0))
        labels.append(torch.tensor(lm_dataset_val[i]["labels"]).unsqueeze(0))
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.cat(labels, dim=0)
    val_dataset = TensorDataset(input_ids, attention_masks, labels)
    end_time = time.time()
    print("Finish loading val dataset, time cost:", str(end_time-start_time))

    train_dataloader = DataLoader(
        train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size
    )
    validation_dataloader = DataLoader(
        val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size
    )

    return train_dataloader, validation_dataloader

def load_pretrained_state_dict():
    model_official = GPT2LMHeadModel.from_pretrained("gpt2")
    state_dict = model_official.state_dict()
    del model_official
    replaced_kv = {}
    for k, v in state_dict.items():
        if (
            "attn.c_attn.weight" in k
            or "attn.c_proj.weight" in k
            or "attn.q_attn.weight" in k
            or "mlp.c_fc.weight" in k
            or "mlp.c_proj.weight" in k
        ):
            replaced_kv[k] = v.transpose(1,0)
        elif "attn.c_attn.bias" in k:
            replaced_kv[k] = v
    for k, v in replaced_kv.items():
        if "attn.c_attn.weight" in k:
            state_dict.pop(k)
            state_dict[k[:-7]+"_q"+k[-7:]] = v[:v.shape[0]//3]
            state_dict[k[:-7]+"_k"+k[-7:]] = v[v.shape[0]//3:v.shape[0]//3*2]
            state_dict[k[:-7]+"_v"+k[-7:]] = v[v.shape[0]//3*2:]
        elif "attn.c_attn.bias" in k:
            state_dict.pop(k)
            state_dict[k[:-5]+"_q"+k[-5:]] = v[:v.shape[0]//3]
            state_dict[k[:-5]+"_k"+k[-5:]] = v[v.shape[0]//3:v.shape[0]//3*2]
            state_dict[k[:-5]+"_v"+k[-5:]] = v[v.shape[0]//3*2:]
        else:
            state_dict[k] = v

    return state_dict

def finetuning(args):

    # Set the seed value all over the place to make this reproducible.
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    block_size = 1024
    train_dataloader, validation_dataloader = build_dataset(block_size, batch_size=args.batch_size)

    # trace gpt2 backbone
    state_dict = load_pretrained_state_dict()
    gpt2_config = GPT2Config()
    model = ModifiedGPT2LMHeadModel(gpt2_config, block_size)
    model.load_state_dict(state_dict)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    epochs = 3
    total_steps = len(train_dataloader) * epochs
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,  # Default value in run_clm.py
        num_training_steps=total_steps,
    )

    criterion = nn.CrossEntropyLoss()
    for epoch_i in range(epochs):
        t0 = time.time()
        total_train_loss = 0
        model.train()
        step = 0
        # train
        for batch in tqdm(train_dataloader):
            # # Progress update every 40 batches.
            # if step % 40 == 0 and not step == 0:
            #     # Calculate elapsed time in minutes.
            #     elapsed = format_time(time.time() - t0)
            #     print(
            #         "  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.".format(
            #             step, len(train_dataloader), elapsed
            #         )
            #     )

            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            model.zero_grad()
            lm_logits = model(input_ids=input_ids, attention_mask=attention_mask)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss/args.gradient_accumulation_steps

            total_train_loss += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if (step+1) % args.gradient_accumulation_steps==0:
                optimizer.step()
                lr_scheduler.step()
            step += 1

        avg_train_loss = total_train_loss / len(train_dataloader)
        training_time = format_time(time.time() - t0)
        print("\n  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch_i,
            },
            "checkpoint.pth.tar",
        )

        validation(model, validation_dataloader, criterion, device)


def postquant(args):
    # Set the seed value all over the place to make this reproducible.
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    block_size=1024
    if args.checkpoint:
        state_dict = torch.load(args.checkpoint)["model"]
    else:
        state_dict = load_pretrained_state_dict()
    gpt2_config = GPT2Config()
    model = ModifiedGPT2LMHeadModel(gpt2_config, block_size)
    model.load_state_dict(state_dict)
    model.to(device)
    

    qconfig = parse_qconfig(args.qconfig)
    qmodel = QuantModel(model, config=qconfig).to(device)
    qmodel.cuda()
    cudnn.benchmark = True

    calib_dataloader, validation_dataloader = build_dataset(block_size, calibration_size=args.calibration_size)

    print("Start calibration with calibration_size =", str(args.calibration_size))
    start_time = time.time()
    qmodel.prepare_calibration()
    cur_size = 0
    for batch in calib_dataloader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        with torch.no_grad():
            qmodel(input_ids=input_ids, attention_mask=attention_mask)
        cur_size += input_ids.shape[0]
        if cur_size >= args.calibration_size:
            break
    qmodel.calc_qparams()
    end_time = time.time()
    print("Calibration finished. Time cost:", str(end_time-start_time), "s")

    qmodel.set_quant(w_quant=True, a_quant=True)
    validation(qmodel, validation_dataloader, nn.CrossEntropyLoss(), device)


def validation(model, dataloader, criterion, device):
    t0 = time.time()
    model.eval()
    total_eval_loss = 0
    for batch in tqdm(dataloader):
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

        with torch.no_grad():
            lm_logits = model(input_ids=input_ids, attention_mask=attention_mask)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        total_eval_loss += loss.item()
    avg_val_loss = total_eval_loss / len(dataloader)
    perplexity = math.exp(avg_val_loss)
    validation_time = format_time(time.time() - t0)
    print("  Perplexity: {0:.2f}".format(perplexity))
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))


def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="sub-command help")

#    a fine-tuning worker
    parser_finetuning = subparsers.add_parser(
        "finetuning", help="the entrance of GPT2 fine-tuning"
    )
    parser_finetuning.set_defaults(func=finetuning)
    parser_finetuning.add_argument("--batch_size", default=4, type=int)
    parser_finetuning.add_argument("--gradient_accumulation_steps", default=4, type=int)


    # a PTQ worker
    parser_postquant = subparsers.add_parser(
        "postquant", help="the entrance of GPT2 post training quantization"
    )
    parser_postquant.set_defaults(func=postquant)
    parser_postquant.add_argument("qconfig")
    parser_postquant.add_argument("--checkpoint", default=None, type=str)
    parser_postquant.add_argument("--calibration_size", default=16, type=int)

    args = parser.parse_args()
    args.func(args)
