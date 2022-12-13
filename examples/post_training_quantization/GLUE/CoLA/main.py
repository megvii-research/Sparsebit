import random
import copy
import os
import time
import datetime
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from model import BertModel, BertForSequenceClassification
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup

from sparsebit.quantization import QuantModel, parse_qconfig


def build_dataset():
    # Load the dataset into a pandas dataframe.
    df = pd.read_csv(
        "./cola_public/raw/in_domain_train.tsv",
        delimiter="\t",
        header=None,
        names=["sentence_source", "label", "label_notes", "sentence"],
    )
    sentences = df.sentence.values
    labels = df.label.values

    print("Loading BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    max_len = 0
    for sent in sentences:
        # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
        input_ids = tokenizer.encode(sent, add_special_tokens=True)
        # Update the maximum sentence length.
        max_len = max(max_len, len(input_ids))
    print("Max sentence length: ", max_len)

    input_ids = []
    attention_masks = []
    for sent in sentences:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=64,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors="pt",  # Return pytorch tensors.
        )
        input_ids.append(encoded_dict["input_ids"])
        attention_masks.append(encoded_dict["attention_mask"])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks, labels)
    # split into train & validation
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    # Divide the dataset by randomly selecting samples.
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # For fine-tuning BERT on a specific task, the authors recommend a batch size of 16 or 32.
    batch_size = 32
    train_dataloader = DataLoader(
        train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size
    )
    validation_dataloader = DataLoader(
        val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size
    )

    return train_dataloader, validation_dataloader


def finetuning(args):

    # Set the seed value all over the place to make this reproducible.
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataloader, validation_dataloader = build_dataset()

    # trace bert backbone
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    bert_model.embeddings.seq_length = 64  # a workaround for torch.fx trace
    bert_model.config.num_labels = 2
    model = BertForSequenceClassification(bert_model, bert_model.config)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    epochs = 4
    total_steps = len(train_dataloader) * epochs
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,  # Default value in run_glue.py
        num_training_steps=total_steps,
    )

    total_t0 = time.time()
    criterion = nn.CrossEntropyLoss()
    for epoch_i in range(0, epochs):
        t0 = time.time()
        total_train_loss = 0
        model.train()
        # train
        for step, batch in enumerate(train_dataloader):
            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                print(
                    "  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.".format(
                        step, len(train_dataloader), elapsed
                    )
                )

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            dummy_token_type_ids = torch.zeros_like(b_input_ids, dtype=torch.long).to(
                device
            )

            model.zero_grad()
            logits = model(b_input_ids, b_input_mask, dummy_token_type_ids)
            loss = criterion(
                logits.view(-1, model.config.num_labels), b_labels.view(-1)
            )

            total_train_loss += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        training_time = format_time(time.time() - t0)
        print("\n  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))

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

    calib_dataloader, validation_dataloader = build_dataset()

    # load checkpoint
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    bert_model.embeddings.seq_length = 64  # a workaround for torch.fx trace
    bert_model.config.num_labels = 2
    model = BertForSequenceClassification(bert_model, bert_model.config)
    model.load_state_dict(torch.load(args.checkpoint)["model"])

    qconfig = parse_qconfig(args.qconfig)
    qmodel = QuantModel(model, config=qconfig).to(device)

    cudnn.benchmark = True

    qmodel.prepare_calibration()
    calibration_size, cur_size = 128, 0
    for batch in calib_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        dummy_token_type_ids = torch.zeros_like(b_input_ids, dtype=torch.long).to(
            device
        )
        with torch.no_grad():
            qmodel(b_input_ids, b_input_mask, dummy_token_type_ids)
        cur_size += b_input_ids.shape[0]
        if cur_size >= calibration_size:
            break
    qmodel.calc_qparams()

    qmodel.set_quant(w_quant=True, a_quant=True)
    validation(qmodel, validation_dataloader, nn.CrossEntropyLoss(), device)

    b_input_ids = b_input_ids.cpu()
    b_input_mask = b_input_mask.cpu()
    dummy_token_type_ids = torch.zeros_like(b_input_ids, dtype=torch.long)
    dummy_data = (b_input_ids, b_input_mask, dummy_token_type_ids)
    qmodel.export_onnx(dummy_data, name="qBERT.onnx")


def validation(model, dataloader, criterion, device):
    t0 = time.time()
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0
    for batch in dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        dummy_token_type_ids = torch.zeros_like(b_input_ids, dtype=torch.long).to(
            device
        )
        with torch.no_grad():
            logits = model(b_input_ids, b_input_mask, dummy_token_type_ids)
            loss = criterion(logits.view(-1, 2), b_labels.view(-1))
        total_eval_loss += loss.item()
        label_ids = b_labels.cpu().numpy()
        total_eval_accuracy += flat_accuracy(logits.detach().cpu().numpy(), label_ids)
    avg_val_accuracy = total_eval_accuracy / len(dataloader)
    avg_val_loss = total_eval_loss / len(dataloader)
    validation_time = format_time(time.time() - t0)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))


def flat_accuracy(preds, labels):
    """
    calculate the accuracy of our predictions vs labels
    """
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


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

    # a fine-tuning worker
    parser_finetuning = subparsers.add_parser(
        "finetuning", help="the entrance of BERT fine-tuning"
    )
    parser_finetuning.set_defaults(func=finetuning)

    # a fine-tuning worker
    parser_postquant = subparsers.add_parser(
        "postquant", help="the entrance of BERT post training quantization"
    )
    parser_postquant.set_defaults(func=postquant)
    parser_postquant.add_argument("qconfig")
    parser_postquant.add_argument("checkpoint")

    args = parser.parse_args()
    args.func(args)
