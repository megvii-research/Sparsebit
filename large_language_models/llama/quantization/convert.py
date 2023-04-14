import os
import math
import time

import torch
import torch.nn as nn
import transformers

from utils.gptq import GPTQ
from utils.modelutils import find_layers
from utils.quant import QuantLinear, Quantizer, make_quant, quantize


def get_llama(model_name):
    import torch

    def skip(*args, **kwargs):
        pass

    from transformers import LlamaForCausalLM

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)

    model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype="auto")
    model.seqlen = 2048
    torch.set_default_dtype(torch.float)
    model.eval()

    return model


def get_wikitext2(nsamples, seed, seqlen, model_name):
    from datasets import load_dataset

    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    import random

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


@torch.no_grad()
def llama_sequential(model, dataloader, dev, means=None, stds=None):
    print("Starting ...")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}
    position_ids = torch.arange(
        0, model.seqlen, dtype=torch.long, device=inps.device
    ).unsqueeze(0)

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:  # calibration
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]

    print("Ready.")

    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)

        subset = find_layers(layer)
        gptq = {}
        for name in subset:
            gptq[name] = GPTQ(subset[name])
            gptq[name].quantizers = []
            for bit in args.candidate_bits:
                quantizer = Quantizer()
                mse_flag = True if bit == 2 else False
                quantizer.configure(bit, perchannel=True, sym=False, mse=mse_flag)
                gptq[name].quantizers.append(quantizer)

        def add_batch(name):
            def tmp(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            outs[j] = layer(
                inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids,
            )[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(i, name)
            print("Quantizing ...")
            bit_idx = gptq[name].fasterquant(
                percdamp=args.percdamp,
                groupsize=args.groupsize,
            )
            quantizer = gptq[name].quantizers[bit_idx]
            quantizer.find_params(
                gptq[name].layer.weight.data, weight=True, groupsize=args.groupsize
            )
            quantizers["model.layers.%d.%s" % (i, name)] = quantizer
            print("model.layers.%d.%s: %d" % (i, name, quantizer.bit))
            gptq[name].free()
        for j in range(args.nsamples):
            outs[j] = layer(
                inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids,
            )[0]

        layers[i] = layer.cpu()
        del gptq
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache

    return quantizers


@torch.no_grad()
def llama_eval(model, testenc, dev, args):
    print("Evaluation...")

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}
    position_ids = torch.arange(
        0, model.seqlen, dtype=torch.long, device=inps.device
    ).unsqueeze(0)

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]

    for i in range(len(layers)):
        layer = layers[i].to(dev)
        for j in range(nsamples):
            outs[j] = layer(
                inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids,
            )[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache


def llama_pack(model, quantizers, groupsize=-1):
    layers = find_layers(model)
    layers_bit = {k: v.bit for k, v in quantizers.items()}
    make_quant(model, layers_bit, groupsize=groupsize)
    qlayers = find_layers(model, [QuantLinear])
    print("Packing ...")
    for name in qlayers:
        print(name)
        quantizers[name] = quantizers[name].cpu()
        qlayers[name].pack(layers[name], quantizers[name].scale, quantizers[name].zero)
    print("Done.")
    return model


def load_qllama(model, checkpoint):
    model = get_llama(model)
    layers = find_layers(model)
    for name in ["lm_head"]:
        if name in layers:
            del layers[name]
    ckpt = torch.load(checkpoint)
    make_quant(model, ckpt["layers_bit"])

    print("Loading Quant model ...")
    model.load_state_dict(ckpt["model"])
    model.seqlen = 2048
    print("Done.")

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_name",
        type=str,
        help="LLaMA model to load; pass `decapoda-research/llama-7b/13b/30b/65b-hf`.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples."
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument(
        "--candidate-bits",
        type=int,
        required=True,
        nargs="+",
        help="the target bit of mixed-precision.",
    )
    parser.add_argument(
        "--groupsize",
        type=int,
        default=-1,
        help="Groupsize to use for quantization; default uses full row.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="",
        help="Save quantized checkpoint under this name.",
    )
    args = parser.parse_args()

    DEV = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # load fp16 model
    model = get_llama(args.model_name)

    # load dataloaders
    dataloader, testloader = get_wikitext2(
        nsamples=args.nsamples,
        seed=args.seed,
        model_name=args.model_name,
        seqlen=model.seqlen,
    )

    # convert
    quantizers = llama_sequential(model, dataloader, DEV)

    # evaluation
    print("The Perplexity on wikiText2: ")
    llama_eval(model, testloader, DEV, args)

    if args.save:
        llama_pack(model, quantizers, groupsize=args.groupsize)
        layers_bit = {k: v.bit for k, v in quantizers.items()}
        torch.save(
            {
                "model": model.state_dict(),
                "hyper_parameters": {"groupsize": args.groupsize},
                "layers_bit": layers_bit,
            },
            args.save,
        )
