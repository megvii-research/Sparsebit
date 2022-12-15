import collections
import numpy as np
from tqdm import tqdm
from functools import partial

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import evaluate
from datasets import load_dataset
from transformers import BertTokenizerFast, default_data_collator, get_scheduler

from model import BertModel, BertForQuestionAnswering
from sparsebit.quantization import QuantModel, parse_qconfig

MAX_LENGTH = 384
STRIDE = 128


def build_train_dataloader(args, raw_datasets):

    def preprocess_training_examples(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=MAX_LENGTH,
            truncation="only_second",
            stride=STRIDE,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer = answers[sample_idx]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label is (0, 0)
            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    tokenizer = BertTokenizerFast.from_pretrained(args.architecture, do_lower_case=True)
    train_dataset = raw_datasets["train"].map(
        preprocess_training_examples,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )
    train_dataset.set_format("torch")
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=args.batch_size,
    )
    return train_dataloader, train_dataset


def build_validation_dataloader(args, raw_datasets):

    def preprocess_validation_examples(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=MAX_LENGTH,
            truncation="only_second",
            stride=STRIDE,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_map = inputs.pop("overflow_to_sample_mapping")
        example_ids = []

        for i in range(len(inputs["input_ids"])):
            sample_idx = sample_map[i]
            example_ids.append(examples["id"][sample_idx])

            sequence_ids = inputs.sequence_ids(i)
            offset = inputs["offset_mapping"][i]
            inputs["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
            ]

        inputs["example_id"] = example_ids
        return inputs

    tokenizer = BertTokenizerFast.from_pretrained(args.architecture, do_lower_case=True)
    validation_dataset = raw_datasets["validation"].map(
        preprocess_validation_examples,
        batched=True,
        remove_columns=raw_datasets["validation"].column_names,
    )
    validation_set = validation_dataset.remove_columns(["example_id", "offset_mapping"])
    validation_set.set_format("torch")
    validation_dataloader = DataLoader(
        validation_set, collate_fn=default_data_collator, batch_size=2 * args.batch_size,
    )
    return validation_dataloader, validation_dataset


def compute_metrics(start_logits, end_logits, features, examples):
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    n_best = 20
    max_answer_length = 30
    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    metric = evaluate.load("squad")
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)


def finetuning(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    raw_datasets = load_dataset('squad')
    train_dataloader, _ = build_train_dataloader(args, raw_datasets)

    bert_model = BertModel.from_pretrained("bert-base-uncased", add_pooling_layer=False)
    bert_model.embeddings.seq_length = MAX_LENGTH
    bert_model.config.num_labels = 2
    model = BertForQuestionAnswering(bert_model, bert_model.config)
    model.cuda()

    total_training_steps = args.epochs * len(train_dataloader)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=total_training_steps,
    )

    progress_bar = tqdm(range(total_training_steps))

    def calc_loss_from_logits(logits, start_positions, end_positions):
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        # If we are on multi-GPU, split add a dimension
        if len(start_positions.size()) > 1:
           start_positions = start_positions.squeeze(-1)
        if len(end_positions.size()) > 1:
            end_positions = end_positions.squeeze(-1)
        # sometimes the start/end positions are outside our model inputs, we ignore these terms
        ignored_index = start_logits.size(1)
        start_positions = start_positions.clamp(0, ignored_index)
        end_positions = end_positions.clamp(0, ignored_index)
        loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        loss = (start_loss + end_loss) / 2
        return loss

    for epoch in range(args.epochs):
        # training
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch_data = {k: v.to(device) for k, v in batch.items() if k not in ["start_positions", "end_positions"]}
            logits = model(**batch_data)
            # calculate loss
            loss = calc_loss_from_logits(logits,
                                         batch["start_positions"].to(device),
                                         batch["end_positions"].to(device))
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
            },
            "checkpoint.pth.tar",
        )
        print("Running evaluation of epoch_{}".format(epoch))
        evaluation(args, model, raw_datasets, device)


def postquant(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    raw_datasets = load_dataset('squad')
    calib_dataloader, _ = build_train_dataloader(args, raw_datasets)
    bert_model = BertModel.from_pretrained("bert-base-uncased", add_pooling_layer=False)
    bert_model.embeddings.seq_length = MAX_LENGTH
    bert_model.config.num_labels = 2
    model = BertForQuestionAnswering(bert_model, bert_model.config)
    model.load_state_dict(torch.load(args.checkpoint)["model"])
    model.to(device)

    qconfig = parse_qconfig(args.qconfig)
    qmodel = QuantModel(model, config=qconfig).to(device)

    cudnn.benchmark = True
    qmodel.prepare_calibration()
    calibration_size, cur_size = 128, 0
    for batch in calib_dataloader:
        batch = {k: v.to(device) for k, v in batch.items() if k not in ["start_positions", "end_positions"]}
        qmodel(**batch)
        cur_size += batch["input_ids"].shape[0]
        if cur_size >= calibration_size:
            break
    qmodel.calc_qparams()

    qmodel.set_quant(w_quant=True, a_quant=True)
    evaluation(args, qmodel, raw_datasets, device)

    # export onnx
    dummy_data = []
    for batch in calib_dataloader:
        dummy_data.append(batch["input_ids"][:1, :])
        dummy_data.append(batch["attention_mask"][:1, :])
        dummy_data.append(batch["token_type_ids"][:1, :])
        break
    qmodel.export_onnx(tuple(dummy_data), name="qBERT.onnx")


def evaluation(args, model, raw_datasets, device):
    dataloader, dataset = build_validation_dataloader(args, raw_datasets)
    model.eval()
    start_logits_list, end_logits_list = [], []
    for batch in tqdm(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            logits = model(**batch)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        start_logits_list.append(start_logits.cpu().numpy())
        end_logits_list.append(end_logits.cpu().numpy())
    # cat all results to evaluate f1-score
    start_logits = np.concatenate(start_logits_list)[: len(dataset)]
    end_logits = np.concatenate(end_logits_list)[: len(dataset)]

    metrics = compute_metrics(
        start_logits, end_logits, dataset, raw_datasets["validation"]
    )
    print(metrics)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="sub-command help")

    # a fine-tuning worker
    parser_finetuning = subparsers.add_parser(
        "finetuning", help="the entrance of BERT fine-tuning"
    )
    parser_finetuning.add_argument("--architecture", type=str, help="the architecture of BERT", default="bert-base-uncased")
    parser_finetuning.add_argument("--batch-size", type=int, default=8)
    parser_finetuning.add_argument("--epochs", type=int, default=3)
    parser_finetuning.add_argument("--lr", type=int, default=2e-5)
    parser_finetuning.set_defaults(func=finetuning)

    parser_postquant = subparsers.add_parser(
        "postquant", help="the entrance of BERT post training quantization"
    )
    parser_postquant.set_defaults(func=postquant)
    parser_postquant.add_argument("qconfig")
    parser_postquant.add_argument("checkpoint")
    parser_postquant.add_argument("--architecture", type=str, help="the architecture of BERT", default="bert-base-uncased")
    parser_postquant.add_argument("--batch-size", type=int, default=8)

    args = parser.parse_args()
    args.func(args)

