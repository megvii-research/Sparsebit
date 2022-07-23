How to register a custom qmodule
==============================================================

Register custom qmodule
------------------------------------------------

In Sparsebit, Models are registed as QModels, with all operators in model registered as QModule, which is easy to be further searched or managed. All Qmodules are and should be defined in ``sparsebit/quantization/modules/``

A typical QModule is registered as follows, here ``QAdaptiveAveragePool`` is used as an example.

.. code-block:: python 
    :linenos:

    @register_qmodule(sources=[nn.AdaptiveAvgPool2d, F.adaptive_avg_pool2d]) #sources are operators to be replaced. All operators in model should be registered as qmodule for easier management.
    class QAdaptiveAvgPool2d(QuantOpr): #Name of QModule
        def __init__(self, org_module, config=None): #org_module is the source operator.
            super().__init__()
            if isinstance(org_module, nn.Module): # for nn.Module operators, their attributes can be accessed directly.
                self.output_size = org_module.output_size
            else: # for functional operators like F.conv, torch.mul, operator.add, their attributes are saved in args. you can print args out for further informations.
                self.output_size = org_module.args[1]
            self._repr_info = "Q" + org_module.__repr__()

        def forward(self, x_in, *args): # for functional operators with extra args, *args must be announced here.
            x_in = self.input_quantizer(x_in)
            out = F.adaptive_avg_pool2d(x_in, self.output_size) # Operators MUST be written as functions to avoid repeated matching
            return out

Add new qmodule to package content
------------------------------------------------------------------------------------------------

To further utilize your custom qmodule, you should add ``from . import YOUR_QMODULE_NAME`` to ``sparsebit/quantization/modules/__init__.py``

Add additional operator fusions
------------------------------------------------------------------------------------------------

Please refer to :doc:`How to register custom fusion <./register_custom_fusion>`