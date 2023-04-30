import os
from functools import partial
import torch
from peft import PeftModel
import transformers
import gradio as gr

from qlora import PeftQModel
from utils import load_qllama, QuantLinear

assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig


def build_model(args):
    if args.load_qlora:
        model_cachedir = "./caches/llama-7b/"
        config = transformers.AutoConfig.from_pretrained(args.llama_config)
        model = load_qllama(config, args.qllama_checkpoint)
        model = PeftQModel.from_pretrained(
            model, args.qlora_dir, torch_dtype=torch.float16
        )
        if torch.cuda.is_available():
            model.cuda()
    else:
        model = LlamaForCausalLM.from_pretrained(
            "decapoda-research/llama-7b-hf",
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            "tloen/alpaca-lora-7b",
            torch_dtype=torch.float16,
            device_map={"": 0},
        )
    return model


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""


def evaluate(
    tokenizer,
    instruction,
    input=None,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    **kwargs,
):
    prompt = generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=2048,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return output.split("### Response:")[1].strip()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load-qlora", action="store_true", help="A flag indicates use quant lora"
    )
    parser.add_argument(
        "--llama-config", type=str, help="the path to save llama config.json"
    )
    parser.add_argument(
        "--qllama-checkpoint", type=str, help="A path to the quantized LLaMa backbone"
    )
    parser.add_argument(
        "--qlora-dir",
        type=str,
        help="a path to the save dir includes adapter_config.json & adapter_model.bin",
    )
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    # all llama models use the same tokenizer
    tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")

    model = build_model(args)
    model.eval()

    evaluate_w_tokenizer = partial(evaluate, tokenizer)
    gr.Interface(
        fn=evaluate_w_tokenizer,
        inputs=[
            gr.components.Textbox(
                lines=2, label="Instruction", placeholder="Tell me about alpacas."
            ),
            gr.components.Textbox(lines=2, label="Input", placeholder="none"),
            gr.components.Slider(minimum=0, maximum=1, value=0.1, label="Temperature"),
            gr.components.Slider(minimum=0, maximum=1, value=0.75, label="Top p"),
            gr.components.Slider(
                minimum=0, maximum=100, step=1, value=40, label="Top k"
            ),
            gr.components.Slider(minimum=0, maximum=4, step=1, value=4, label="Beams"),
        ],
        outputs=[
            gr.inputs.Textbox(
                lines=5,
                label="Output",
            )
        ],
        title="🦙🌲 Alpaca-LoRA",
        description="Alpaca-LoRA is a 7B-parameter LLaMA model finetuned to follow instructions. It is trained on the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) dataset and makes use of the Huggingface LLaMA implementation. For more information, please visit [the project's website](https://github.com/tloen/alpaca-lora).",
    ).launch(server_name="0.0.0.0", server_port=args.port, share=True)
