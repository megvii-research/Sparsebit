import os
import warnings
import torch
import transformers

from transformers import LlamaForCausalLM
from utils.llama_wrapper import LLaMAClass
from utils.modelutils import find_layers
from utils.quant import make_quant


def load_llama(model_name, load_quant=True, config=None, checkpoint="", groupsize=-1):
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)

    print("Loading model... ", end="")
    if not load_quant:
        model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype="auto")
        model.seqlen = 2048
    else:
        assert os.path.exists(checkpoint), "loading low-bit model requires checkpoint"
        model = LLaMAClass(config)

    torch.set_default_dtype(torch.float)
    model.eval()

    if load_quant:
        layers = find_layers(model)
        for name in ["lm_head"]:
            if name in layers:
                del layers[name]
        ckpt = torch.load(checkpoint)
        if ckpt["hyper_parameters"]["groupsize"] != groupsize:
            warnings.warn("Quantization group-size not-set / mismatch detected.")
        make_quant(model, ckpt["layers_bit"], groupsize=groupsize)
        print("Loading Quant model ...")
        model.load_state_dict(ckpt["model"])
        model.seqlen = 2048
    print("done.")
    return model


def inference(args):
    DEV = torch.device("cuda:0")
    prompt = "Let me tell you a story:"
    # prompt = "why is the sky blue?"

    config = transformers.AutoConfig.from_pretrained(args.config_cache)
    model = load_llama(
        args.model,
        load_quant=True,
        config=config,
        checkpoint=args.checkpoint,
        groupsize=args.groupsize,
    )
    model.single_device_mode = bool(args.single_device_mode)
    if not model.single_device_mode:
        model.to(DEV)

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer_cache)
    import time

    inputs = tokenizer.encode(prompt, return_tensors="pt").to(DEV)
    start_time = time.time()
    outputs = model.generate(inputs, max_new_tokens=100, do_sample=False)
    end_time = time.time()
    print("time = {} s.".format(end_time - start_time))
    print(tokenizer.decode(outputs[0]))


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model", type=str, help="LLaMA model to load; pass `llama-7b/13b/30b/65b`."
    )
    parser.add_argument(
        "checkpoint",
        type=str,
        default="",
        help="a checkpoint path from local storage",
    )
    parser.add_argument(
        "--config_cache",
        type=str,
        default="",
        required=True,
        help="config from local storage",
    )
    parser.add_argument(
        "--tokenizer_cache",
        type=str,
        default="",
        required=True,
        help="tokenizer config from local storage",
    )
    parser.add_argument(
        "--groupsize",
        type=int,
        default=-1,
        help="Groupsize to use for quantization; default uses full row.",
    )
    parser.add_argument(
        "--single_device_mode",
        default=False,
        action="store_true",
        help="run LLaMA large models with single device",
    )
    args = parser.parse_args()
    inference(args)


main()
