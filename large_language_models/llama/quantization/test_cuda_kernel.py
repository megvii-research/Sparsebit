"""
Tests for GPTQ quantization implementation.
inp1 x inp2 ==> out
inp1: (B, M) || (B, C, M)
inp2: (M, N)
out : (B, N) || (B, C, N)
"""

import torch
import torch.nn as nn

from utils.quant import Quantizer, quantize, QuantLinear


def init():
    assert torch.cuda.is_available(), "CUDA is required to run tests"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def run_case(bit, B, M, N, C=None, dtype=torch.float32, GS=-1):
    assert bit in [2, 3, 4]
    assert GS == -1 or GS % ({2: 64, 3: 128, 4: 128}[bit]) == 0
    init()
    layer = nn.Linear(M, N)
    vec = torch.randn((B, M) if C is None else (B, C, M)).to("cuda")
    quantizer = Quantizer()

    quantizer.configure(bit=bit, perchannel=True, sym=False, mse=False)
    quantizer.find_params(layer.weight.data, weight=True, groupsize=GS)
    layer.weight.data = quantize(
        layer.weight.data.view(-1, M if GS == -1 else GS),
        quantizer.scale.view(-1, 1),
        quantizer.zero.view(-1, 1),
        quantizer.maxq,
    ).view(N, M)

    qlayer = QuantLinear(layer.in_features, layer.out_features, bit=bit, groupsize=GS)
    qlayer.pack(layer, quantizer.scale, quantizer.zero)

    qlayer = qlayer.to("cuda")
    layer = layer.to("cuda")

    with torch.no_grad():
        gt_out = layer(vec)
        sim_out = qlayer(vec)
        torch.testing.assert_allclose(sim_out, gt_out, rtol=1e-5, atol=1e-5)


def test_OPT_175B_FC2_matvec():
    # derived from official GPTQ tests
    run_case(bit=2, B=1, M=12288, N=12288 * 4)
    run_case(bit=3, B=1, M=12288, N=12288 * 4)
    run_case(bit=4, B=1, M=12288, N=12288 * 4)


def test_regular_FC():
    run_case(bit=2, B=1, M=8192, N=8192 * 4)
    run_case(bit=3, B=1, M=9216, N=9216 * 4)
    run_case(bit=4, B=1, M=8192, N=8192 * 4)


def test_irregular_FC():
    run_case(bit=2, B=1, M=6661, N=25163)
    run_case(bit=3, B=1, M=6661, N=25163)
    run_case(bit=4, B=1, M=6661, N=25163)


def test_single_block_regular_FC():
    run_case(bit=2, B=1, M=1024, N=1024)
    run_case(bit=3, B=1, M=1024, N=1024)
    run_case(bit=4, B=1, M=128, N=64)


def test_single_block_irregular_FC():
    run_case(bit=2, B=1, M=719, N=857)
    run_case(bit=3, B=1, M=719, N=857)
    run_case(bit=4, B=1, M=127, N=61)


def test_multibatch_OPT_127B_FC2_matvec():
    run_case(bit=2, B=32, M=12288, N=12288 * 4)
    run_case(bit=3, B=32, M=12288, N=12288 * 4)
    run_case(bit=4, B=32, M=12288, N=12288 * 4)


def test_multibatch_regular_FC():
    run_case(bit=2, B=29, M=8192, N=8192 * 4)
    run_case(bit=3, B=29, M=8192, N=8192 * 4)
    run_case(bit=4, B=29, M=8192, N=8192 * 4)


def test_multibatch_irregular_FC():
    run_case(bit=2, B=31, M=6661, N=25163)
    run_case(bit=3, B=31, M=6661, N=25163)
    run_case(bit=4, B=31, M=6661, N=25163)


def test_multibatch_1token_FC():
    run_case(bit=2, B=32, C=1, M=6661, N=25163)
    run_case(bit=3, B=32, C=1, M=6661, N=25163)
    run_case(bit=4, B=32, C=1, M=6661, N=25163)


def test_multibatch_8token_FC():
    run_case(bit=2, B=4, C=8, M=6661, N=25163)
    run_case(bit=3, B=4, C=8, M=6661, N=25163)
    run_case(bit=4, B=4, C=8, M=6661, N=25163)


def test_OPT_175B_FC2_matvec_groupsize_min():
    run_case(bit=2, B=1, M=12288, N=12288 * 4, GS=64)
    run_case(bit=3, B=1, M=12288, N=12288 * 4, GS=128)
    run_case(bit=4, B=1, M=12288, N=12288 * 4, GS=128)


def test_multibatch_regular_FC_groupsize_min():
    run_case(bit=2, B=29, M=8192, N=8192 * 4, GS=64)
    run_case(bit=3, B=29, M=8192, N=8192 * 4, GS=128)
    run_case(bit=4, B=29, M=8192, N=8192 * 4, GS=128)


def test_groupsize_3x():
    run_case(bit=2, B=4, M=6144, N=6144 * 4, GS=192)
    run_case(bit=3, B=4, M=6144, N=6144 * 4, GS=384)
    run_case(bit=4, B=4, M=6144, N=6144 * 4, GS=384)
