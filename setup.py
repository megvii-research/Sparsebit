import setuptools


def read_requirements():
    reqs = []
    with open("requirements.txt", "r") as fin:
        for line in fin.readlines():
            reqs.append(line.strip())
    return reqs


setuptools.setup(
    name="sparsebit",
    version="0.0.1",
    author="spb",
    description="A toolkit for pruning and quantization.",
    packages=setuptools.find_packages(),
    install_requires=read_requirements(),
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    package_data={
        "quantization": [
            "quantization/torch_extensions/*.c",
            "quantization/torch_extensions/*.cpp",
            "quantization/torch_extensions/*.h",
            "quantization/torch_extensions/*.hpp",
            "quantization/torch_extensions/*.cu",
            "quantization/torch_extensions/*.cuh",
        ]
    },
    python_requires=">=3.6",
)
