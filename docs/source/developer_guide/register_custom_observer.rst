How to register a custom observer
==============================================================

To register a custom observer, you need to create a new file in ``sparsebit/quantization/observers/`` and add it to ``sparsebit/quantization/observers/__init__.py``

Register custom observer
------------------------------------------------

An observer example is shown below. ``calc_minmax`` should be overwritten by your method for range calculation.

.. code-block:: python 
    :linenos:

    @register_observer
    class Observer(BaseObserver):
        TYPE = "minmax" # Observer name

        def __init__(self, config, qdesc):
            super(Observer, self).__init__(config, qdesc)

        def calc_minmax(self): #how to calculate minmax value for quantizer, must be overwritten.
            # note: the data in data_cache is c_first
            assert (
                len(self.data_cache) > 0
            ), "Before calculating the quant params, the observation of data should be done"
            data = torch.cat(self.data_cache, axis=1)
            self.reset_data_cache()
            ...
            return self.min_val, self.max_val


Add new observer to package content
------------------------------------------------------------------------------------------------

To further utilize your custom observer, you should add ``from . import YOUR_OBSERVER_NAME`` to ``sparsebit/quantization/observers/__init__.py``
