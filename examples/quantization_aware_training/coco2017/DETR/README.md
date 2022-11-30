# DETR QAT example

## preparation

The `DETR` pretrained model is the checkpoint from https://github.com/facebookresearch/detr . The example will automatically download the checkpoint using `torch.hub.load`.

The datasets used in this example are train dataset and validation dataset of COCO2017. They can be downloaded from http://cocodataset.org. also the relative cocoapi should be installed.

## Usage

```shell
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py qconfig_lsq_8w8f.yaml --coco_path /path/to/coco
```

## Metrics

|DETR-R50|mAPc|AP50|AP75| remarks|
|-|-|-|-|-|
|Float|0.421|0.623|0.443|baseline|
|8w8f|
