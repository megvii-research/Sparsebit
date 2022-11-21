## How to get the dataset?
- first: `download ImageNet from https://image-net.org/download-images`
- second: `run a script from https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh`

## Training
```
python3 -m torch.distributed.launch --nproc_per_node=4 --use_env main.py qconfig.yaml [imagenet-folder with train and val folders] -a [an architecture in torchvision.models] --output_dir [the path of log] --pretrained --lr 0.01
```
## Evaluation
```
python3 main.py qconfig.yaml [imagenet-folder with train and val folders] -a [an architecture in torchvision.models] --output_dir --output_dir [the path of log] --eval --resume [the path of your checkpoint]
```
## ImageNet Benchmark
- Task: ImageNet
- Eval data num: 50k
- Calib data num: 256
- Weight bit: 4
- Feature bit: 4
- First Layer: 8w
- Last Layer: 8w8f

|model| float | DoReFa|PACT|LSQ| LSQ+ | 
|---|---|---|---|---| --- | 
|ResNet-18| 69.758 |  69.231 | 69.596 | 70.314 [log](https://drive.google.com/file/d/1FDRiOy4xsRVCQJuqhKPmuNmWok2wRmDs/view?usp=share_link) [ckpt](https://drive.google.com/file/d/1UAo1RAa4DAjVW3aj-1y66wpGNRlhKKd4/view?usp=share_link) | - |
|ResNet-50| 76.130 |75.552|76.022| 76.664 [log](https://drive.google.com/file/d/1C4NSOOmEtpJOOvQ_h6QTUIoC4YCZ5h_x/view?usp=share_link) [ckpt](https://drive.google.com/file/d/1Wcvb4ObVQFKcibxV2BFRy5Vjz1n9jQgM/view?usp=share_link) | - | 
|MobileNetV2| 72.154|  68.660 | -  | - | - | 
|RegNetX-600MF|75.034| N/C | 73.578 | - | - | 
|Efficientnet\_b0 | - | - | - | - | 
|Efficientnet\_lite0 | - | - | - | - |
