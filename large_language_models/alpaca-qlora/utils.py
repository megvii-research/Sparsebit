import os
import numpy as np
import torch
import torch.nn as nn
import transformers
from transformers import LlamaForCausalLM
from qmatmul import Quant4Matmul


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=""):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res


def modulename_remap(ckpt):
    new_state_dict = {}
    for k, v in ckpt.items():
        new_name = k.replace(".decoder", "")
        if "feed_forward" in new_name:
            new_name = new_name.replace("feed_forward", "mlp")
            if "w1" in new_name:
                new_name = new_name.replace("w1", "gate_proj")
            elif "w2" in new_name:
                new_name = new_name.replace("w2", "down_proj")
            elif "w3" in new_name:
                new_name = new_name.replace("w3", "up_proj")
        if "attention_norm" in new_name:
            new_name = new_name.replace("attention_norm", "input_layernorm")
        if "ffn_norm" in new_name:
            new_name = new_name.replace("ffn_norm", "post_attention_layernorm")
        new_state_dict[new_name] = v
    return new_state_dict


def load_qllama(config, checkpoint=""):
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)

    assert os.path.exists(checkpoint), "loading low-bit model requires checkpoint"
    model = LlamaForCausalLM(config)
    torch.set_default_dtype(torch.float)
    model.eval()

    layers = find_layers(model)
    for name in ["lm_head"]:
        if name in layers:
            del layers[name]
    ckpt = torch.load(checkpoint)["model"]
    ckpt_remapped = modulename_remap(ckpt)
    layers_bit = {k: 4 for k in layers}
    make_quant(model, layers_bit)
    model.load_state_dict(ckpt_remapped)
    model.seqlen = 2048
    print("Loading Model Done")
    return model


def ceiling_div(x, y):
    return (x + y - 1) // y


class QuantLinear(nn.Module):
    def __init__(self, infeatures, outfeatures, bit=4):
        super().__init__()
        # 3 int32 solved at the same time for 3bit
        # 1 int32 solved at the same time for 4bit
        self.register_buffer("zeros", torch.zeros((outfeatures, 1)))
        self.register_buffer("scales", torch.ones((outfeatures, 1)))
        self.register_buffer("bias", torch.zeros(outfeatures))
        self.bit = bit
        self.parallel_bit_nums = 1 if self.bit in [2, 4] else 3
        assert self.bit in [2, 3, 4], "only support 2/3/4 bit now"
        pack_bits = 8  # 32
        self.register_buffer(
            "qweight",
            torch.zeros(
                (
                    ceiling_div(
                        infeatures * self.bit, pack_bits * self.parallel_bit_nums
                    )
                    * self.parallel_bit_nums,
                    outfeatures,
                ),
                dtype=torch.int8,
            ),
        )
        self.in_features = infeatures
        self.out_features = outfeatures

    def reset_parameters(self):
        pass

    def pack(self, linear, scales, zeros):
        self.zeros = zeros * scales
        self.scales = scales.clone()
        if linear.bias is not None:
            self.bias = linear.bias.clone()

        intweight = torch.round((linear.weight.data + self.zeros) / self.scales).to(
            torch.int
        )
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)
        qweight = np.zeros(
            (
                ceiling_div(intweight.shape[0] * self.bit, 32 * self.parallel_bit_nums)
                * self.parallel_bit_nums,
                intweight.shape[1],
            ),
            dtype=np.uint32,
        )
        i = 0
        row = 0
        H, W = intweight.shape
        while row < qweight.shape[0] and i < H:
            if self.bit == 2:
                for j in range(i, i + 16):
                    if j < H:
                        qweight[row] |= intweight[j] << (2 * (j - i))
                i += 16
                row += 1
            if self.bit == 4:
                for j in range(i, i + 8):
                    if j < H:
                        qweight[row] |= intweight[j] << (4 * (j - i))
                i += 8
                row += 1
            elif self.bit == 3:
                for j in range(i, i + 10):
                    if j < H:
                        qweight[row] |= intweight[j] << (3 * (j - i))
                i += 10

                if i < H:
                    qweight[row] |= intweight[i] << 30
                    row += 1
                    qweight[row] |= (intweight[i] >> 2) & 1
                    i += 1
                else:
                    break

                for j in range(i, i + 10):
                    if j < H:
                        qweight[row] |= intweight[j] << (3 * (j - i) + 1)
                i += 10

                if i < H:
                    qweight[row] |= intweight[i] << 31
                    row += 1
                    qweight[row] |= (intweight[i] >> 1) & 0x3
                    i += 1
                else:
                    break

                for j in range(i, i + 10):
                    if j < H:
                        qweight[row] |= intweight[j] << (3 * (j - i) + 2)
                i += 10
                row += 1

        qweight = qweight.astype(np.int32)
        qweight = qweight.tobytes()
        qweight = np.frombuffer(qweight, dtype="int8")
        qweight = (
            qweight.reshape(self.in_features // 8, self.out_features, 4)
            .transpose(0, 2, 1)
            .reshape(self.in_features // 2, self.out_features)
        )
        self.qweight = torch.from_numpy(qweight)

    def unpack(self, bit):
        assert bit == 4
        from load_cuda_kernel import cuda_kernel

        return (
            cuda_kernel.unpack(
                self.qweight.cuda(),
                (self.zeros / self.scales).round().char().cuda(),
                False,
            )
            * self.scales.t().cuda()
        ).to(self.qweight.device)

    def prepare_backward_scales(self):
        assert self.bit == 4
        w_fake = self.unpack(bit=4)
        w_max = torch.max(w_fake, dim=1).values
        w_min = torch.min(w_fake, dim=1).values
        # use 8bit quantization
        qbit = 8
        mx = 254  # (-127, 127)
        mid = 127
        ic_scales = (w_max - w_min) / mx
        ic_zeros = torch.round(mx - (w_max / ic_scales)) - mid
        ic_zeros *= ic_scales
        self.register_buffer("backward_ic_scales", ic_scales)
        self.register_buffer("backward_ic_zeros", ic_zeros)

    def undo_prepare_backward_scales(self):
        if hasattr(self, "backward_ic_scales") and hasattr(self, "backward_ic_zeros"):
            del self.backward_ic_scales
            del self.backward_ic_zeros

    def train(self, mode: bool = True):
        if not self.training and mode:
            self.prepare_backward_scales()
        elif self.training and not mode:
            self.undo_prepare_backward_scales()
        super().train(mode)

    def forward(self, x):
        f32 = lambda x: x.to(torch.float32)
        result = (
            {4: Quant4Matmul}[self.bit]
            .apply(
                f32(x),
                self.qweight,
                f32(self.scales),
                self.zeros,
                f32(self.bias),
                -1,
                getattr(self, "backward_ic_scales", None),
                getattr(self, "backward_ic_zeros", None),
            )
            .to(x.dtype)
        )

        return result


def make_quant(module, layers_bit, name=""):
    if isinstance(module, QuantLinear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + "." + attr if name != "" else attr
        if name1 in layers_bit:
            setattr(
                module,
                attr,
                QuantLinear(tmp.in_features, tmp.out_features, bit=layers_bit[name1]),
            )
    for name1, child in module.named_children():
        make_quant(child, layers_bit, name + "." + name1 if name != "" else name1)
