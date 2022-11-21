How to register a custom quantizer
==============================================================

To register a custom quantizer, you need to create a new file in ``sparsebit/quantization/quantizers/`` and add it to ``sparsebit/quantization/quantizers/__init__.py``

Register custom quantizer
------------------------------------------------

A quantizer example is shown below. ``calc_qparam`` is used for parameter initialization, and ``_forward`` includes quantization operations.

.. code-block:: python 
    :linenos:

    @register_quantizer
    class Quantizer(BaseQuantizer):
        TYPE = "LSQ" # quantizer name

        def __init__(self, config):
            super(Quantizer, self).__init__(config)
            ...

        def calc_qparams(self): # if need calibration
            x_oc = self.observer.data_cache.get_data_for_calibration(Granularity.CHANNELWISE)
            self.observer.data_cache.reset()
            ...
            return self.scale, self.zero_point

        def _forward(self, x): # operations for quantization
            ...
            return x_dq

Add new quantizer to package content
------------------------------------------------------------------------------------------------

To further utilize your custom quantizer, you should add ``from . import YOUR_QUANTIZER_NAME`` to ``sparsebit/quantization/quantizers/__init__.py``
