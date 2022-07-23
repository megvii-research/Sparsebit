Get Started
==============================================

Setup Sparsebit
------------------------------------------------------------------------

**0**. **Install CUDA, CuDNN and TensorRT**

Please refer to https://developer.nvidia.com/cuda-toolkit

**1**. **Install Sparsebit**

.. code-block:: shell
    :linenos:

    git clone https://github.com/megvii-research/Sparsebit.git
    cd sparsebit
    python3 setup.py develop --user

**2**. **Install python-TensorRT package**

.. code-block:: shell
    :linenos:

    cd $PATH_TO_TENSORRT/python
    pip3 install tensorrt-x.x.x.x-cp3x-none-linux_x86_64.whl

Quantization Examples
------------------------------------------------------------------------

Two quantization examples are provided. Please refer to the following folds for more details.

- PTQ example: ``$PATH_TO_SPARSTBIT/examples/cifar10_ptq/``

- QAT example: ``$PATH_TO_SPARSTBIT/examples/cifar10_qat/``