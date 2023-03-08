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
    import os
    from torch.utils.cpp_extension import load

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
            os.path.join(basedir, "cuda_kernel_3bit.cu"),
            os.path.join(basedir, "cuda_kernel_2bit.cu"),
        ],
        with_cuda=True,
        build_directory=os.path.join(basedir, "build"),
        extra_cflags=["-O3"],
    )
    print("done.")
    return cuda_kernel


cuda_kernel = load_kernel()
