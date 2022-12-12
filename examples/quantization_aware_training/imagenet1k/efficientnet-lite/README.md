## How to get the dataset?
- first: `download ImageNet from https://image-net.org/download-images`
- second: `run a script from https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh`

## Download float checkpoint
- download checkpoint from https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/lite

## Training
```
python3 -m torch.distributed.launch --nproc_per_node=4 --use_env main.py qconfig_lsq.yaml [imagenet-folder with train and val folders] -a efficientnet_lite0 --output_dir [the path of log] --pretrained --pretrain_path [float checkpoint path] --lr 0.01
```
## Evaluation
```
python3 main.py qconfig_lsq.yaml [imagenet-folder with train and val folders] -a efficientnet_lite0 --output_dir [the path of log] --eval --resume [the path of your checkpoint]
```
## ImageNet Benchmark
- Task: ImageNet
- Eval data num: 50k
- Calib data num: 256
- Weight bit: 4
- Feature bit: 4
- First Layer: 8w8f
- Last Layer: 8w8f

|model| float | DoReFa|PACT|LSQ|
|---|---|---|---|---| --- |
|Efficientnet\_lite0 | 75.1 | - | 65.76 | 72.89 |
