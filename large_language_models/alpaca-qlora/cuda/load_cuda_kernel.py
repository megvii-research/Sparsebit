def requirements_check():
    try:
        import ninja
    except ModuleNotFoundError:
        print(
            '\033[31mNo module "ninja" found. \
            Use "pip install ninja" before running cuda compile.\033[0m'
        )
        assert False, 'install python module "ninja" first'


def load_kernel():
    import torch
    import os
    from torch.utils.cpp_extension import load

    compute_capability = torch.cuda.get_device_capability()
    cuda_arch = compute_capability[0] * 100 + compute_capability[1] * 10

    requirements_check()
    print("Loading cuda kernel... ", end="")
    basedir = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists(os.path.join(basedir, "build")):
        os.makedirs(os.path.join(basedir, "build"))
    cuda_kernel = load(
        name="cuda_kernel",
        sources=[
            os.path.join(basedir, "cuda_kernel.cpp"),
            os.path.join(basedir, "cuda_kernel_4bit.cu"),
            os.path.join(basedir, "int8gemm.cu"),
            os.path.join(basedir, "unpack.cu"),
            os.path.join(basedir, "tokenwise_quant.cu"),
        ],
        with_cuda=True,
        build_directory=os.path.join(basedir, "build"),
        extra_ldflags=[
            "-lcublas_static",
            "-lcublasLt_static",
            "-lculibos",
            "-lcudart",
            "-lcudart_static",
            "-lrt",
            "-lpthread",
            "-ldl",
            "-L/usr/lib/x86_64-linux-gnu/",
        ],
        extra_cflags=["-std=c++17", "-O3"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            f"-DCUDA_ARCH={cuda_arch}",
        ],
    )
    print("done.")
    return cuda_kernel


cuda_kernel = load_kernel()
