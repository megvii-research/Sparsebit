Learn Sparsebit quantization configuration
==============================================================

In Sparsebit, YAML is applied as the experimental management method.

A YAML example is shown below:

.. code-block:: yaml 
    :linenos:

    
    BACKEND: virtual
    W: # Weight quantization options
        QSCHEME: per-channel-symmetric 
        QUANTIZER:
            TYPE: uniform # custom weight quantizer
            BIT: 8 # custom weight bit
        OBSERVER:
            TYPE: MINMAX # custom weight observer
    A: # Activation quantization options
        QSCHEME: per-tensor-affine # custom activation granularity and pattern
        QUANTIZER:
            TYPE: uniform # custom activation quantizer
            BIT: 8 # custom activation bit
        OBSERVER:
            TYPE: MINMAX # custom activation observer

Key word explanation:

- ``BACKEND``: The runtime you want to deploy your model. Currently support:

  - ``virtual`` (for academic)
  - ``onnxruntime``
  - ``tensorrt``

- ``QSCHEME``: Granularity and pattern for quantization. Currently support: 

  - ``per-tensor-symmetric``
  - ``per-tensor-affine``
  - ``per-channel-symmetric``
  - ``per-channel-affine``

- ``QUANTIZER TYPE``: Method for quantization. Currently support:

  - ``uniform``
  - ``DoReFa``
  - ``LSQ``
  - ``LSQ_plus``

- ``OBSERVER TYPE``: Method for observing. Currently support:

  - MinMax