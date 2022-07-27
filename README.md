## Introduction
Sparsebit is a toolkit with pruning and quantization capabilities. It is designed to help researchers compress and accelerate the network by modifying only a few codes in existing pytorch project.

## Quantization 
Quantization turns full-precision params into low-bit precision params, which can compress and accelerate the model without changing its structure. This toolkit supports two common quantization paradigms, Post-Training-Quantization and Quantization-Aware-Training, with following features:

- Benefiting from the support of torch.fx, Sparsebit operates on a QuantModel, and each operation becomes a QuantModule.
- Sparsebit can easily be extended by users to accommodate their own researches. Users can register to extend important objects such as QuantModule, Quantizer and Observer by themselves.
- Exporting [QDQ-ONNX](https://onnxruntime.ai/docs/tutorials/mobile/helpers/#qdq-format-model-helpers) is supported, which can be loaded and deployed by backends such as TensorRT and OnnxRuntime.

## Pruning
About to released.

## Resources
### Documentations
Detailed usage and development guidance is located in the document. Any users in need can refer to it. [docs]()

### CV-Master
- We maintain a public course on quantification at Bilibili, introducing the basics of quantification and our latest work. Interested users can join the course.[video](https://www.bilibili.com/video/BV13a411p7PC?p=1&vd_source=f746210dbb726509198fbec99dfe7367)
- Aiming at better enabling users to understand and apply the knowledge related to model compression, we designed related homework based on Sparsebit. Interested users can complete it by themselves.[quantization\_homework](https://github.com/megvii-research/Sparsebit/blob/homeworks/homeworks/quant_homework.md)

## Join Us
- Our team is always recruiting interns. The required research interests include but are not limited to: model quantification, sparsity and pruning, model distillation, self-supervised learning, model deployment, etc.
- Candidates interested in model compression and acceleration can submit resumes to: sunpeiqin@megvii.com

## Acknowledgement
The implementation of Sparsebit was inspired by several open source projects. We are grateful for these excellent projects and list them as follows:
- [torch](https://github.com/pytorch/pytorch/tree/master/torch/quantization)
- [pytorch-quantization](https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization)
- [PPQ](https://github.com/openppl-public/ppq)
- [MQBench](https://github.com/ModelTC/MQBench)


## License
Sparsebit is released under the Apache 2.0 license.
