## [English](https://github.com/megvii-research/Sparsebit/blob/main/README.md)
## Introduction
Sparsebit是一个具备prune, quantization两个功能的工具包, 其目的是协助研究人员在已有的pytorch工程代码中通过少量的代码修改实现网络的压缩与加速. 

## Quantization 
Quantization是一种将full-precision params转化为low-bit precision params的方法, 可以在不改变模型结构的情况下实现模型的压缩与加速. 工具包支持Post-Training-Quantization和Quantization-Aware-Training两种常用量化范式, 具备如下特点:
- 得益于torch.fx的支持, 以QuantModel为操作对象, 每个operation成为一个QuantModule
- 方便用户扩展. 可自行通过注册扩展 QuantModule, Quantizer和Observer等重要对象, 以满足研究所需
- 支持导出[QDQ-ONNX](https://onnxruntime.ai/docs/tutorials/mobile/helpers/#qdq-format-model-helpers), 可以被tensorrt/onnxruntime等后端加载部署.

### 测试结果
- PTQ results on ImageNet-1k: [link](https://github.com/megvii-research/Sparsebit/blob/main/examples/post_training_quantization/imagenet1k/basecase/README.md)
- QAT results on ImageNet-1k: [link](https://github.com/megvii-research/Sparsebit/blob/main/examples/quantization_aware_training/imagenet1k/README.md)

## Sparse
Sparse在深度学习当中常用于指代减少网络参数或者网络计算量等操作,是个较为宽泛的概念. 目前工具箱支持的Sparse主要面向Pruning操作, 具备如下特点:
- 支持两种剪枝类型: 结构化/非结构化;
- 支持多种操作对象包括: 权重/激活值/模型块/模型层等; 
- 支持多种剪枝算法: L1-norm/L0-norm/Fisher-pruning/Hrank/Slimming...
- 方便用户扩展, 只需要通过定义Sparser即可实现插入自定义剪枝算法的目的
- 支持导出剪枝后的onnx模型, 导出过程综合考虑了结构化剪枝后对应修改的模型结构

## Resources
### Documentations
我们在文档中提供详尽的使用指导和开发指导, 有需要的用户可以自行参考. [docs](https://sparsebit.readthedocs.io/en/latest/)

### CV-Master
- 我们在Bilibili维护了一门关于量化的公开课, 介绍量化的基本知识和小组最新的工作. 有兴趣的用户可以前往. [video](https://www.bilibili.com/video/BV13a411p7PC?p=1&vd_source=f746210dbb726509198fbec99dfe7367)
- 为了更好让大家理解和应用压缩的相关知识, 我们基于Sparsebit设计相关作业, 有兴趣的同学可以自行完成. [quantization\_homework](https://github.com/megvii-research/Sparsebit/blob/homeworks/homeworks/quant_homework.md)

### 重写计划
* [ ] https://github.com/megvii-research/SSQL-ECCV2022
* [ ] https://github.com/megvii-research/FQ-ViT

## Join Us
- 小组常年招收实习生, 包括但不限于: 模型量化, 模型稀疏与剪枝, 模型蒸馏, 自监督学习, 模型部署等.
- 有志于从事模型压缩与加速的同学可以投递简历至: sunpeiqin@megvii.com,

## Acknowledgement
Sparsebit的实现受到以下多个开源工程的启发, 非常感谢这些优秀项目. 我们列举如下:
- [torch](https://github.com/pytorch/pytorch/tree/master/torch/quantization)
- [pytorch-quantization](https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization)
- [PPQ](https://github.com/openppl-public/ppq)
- [MQBench](https://github.com/ModelTC/MQBench)


## License
Sparsebit is released under the Apache 2.0 license.
