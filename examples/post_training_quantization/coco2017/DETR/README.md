# DETR PTQ example

## preparation

The `DETR` pretrained model is the checkpoint from https://github.com/facebookresearch/detr . The example will automatically download the checkpoint using `torch.hub.load`.

The datasets used in this example are train dataset and validation dataset of COCO2017. They can be downloaded from http://cocodataset.org. also the relative cocoapi should be installed.

## Usage

```shell
python3 main.py qconfig.yaml --coco_path /path/to/coco
```
Since mask is not well supported by onnx, we removed mask-related codes and assign the batch size to be 1 only. Dynamic_axes for onnx is also not supported yet.

## Metrics

|DETR-R50|mAPc|AP50|AP75| remarks|
|-|-|-|-|-|
|float|0.421 | 0.623 | 0.443 | baseline
|8w8f|0.332|0.588|0.320| minmax observer|
|8w8f|0.404|0.612|0.421| minmax observer, float w&f for last 2 bbox embed layers|
|8w8f|0.384|0.598|0.402| minmax observer, apply aciq laplace observer for last bbox embed layer|
|8w8f|0.398|0.609|0.420| minmax observer, apply aciq laplace observer for last 2 bbox embed layer|

TRT DETR w/ fixed input shape, enable int8&fp16 QPS: 118.334 on Nvidia 2080Ti. For detailed visualization, please refer to 
```shell
examples/post_training_quantization/coco2017/DETR/DETR_8w8f_visualization_mAP0395.svg
```