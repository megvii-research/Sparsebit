## Training
```
python3 -m torch.distributed.launch --nproc_per_node=4 --use_env main.py qconfig.yaml [imagenet-folder with train and val folders] -a [an architecture in torchvision.models] --multiprocessing-distributed -b 256 --lr 0.01 --print-freq 100 --pretrained
```

## How to get the dataset?
- first: `download ImageNet from https://image-net.org/download-images`
- second: `run a script from https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh`

## ImageNet Benchmark
- Task: ImageNet
- Eval data num: 50k
- Calib data num: 256
- Weight bit: 4
- Feature bit: 4
- First Layer: 8w
- Last Layer: 8w8f
- Backend: TensorRT (symmetric feature quantization)

|model| float | DoReFa|PACT|LSQ|
|---|---|---|---|---|
|ResNet-18| 69.758 |  69.231 | 69.596 | 70.124 |
|ResNet-50| 76.130 |75.552|76.022| 76.362 | 
|MobileNetV2| 72.154|  68.660 | -  | 69.125|
|RegNetX-600MF|75.034|N/C|73.578|-|
| DeiT-tiny |72.2 | - | - | 71.27 |
| Swin-tiny |81.148 | - | - | 80.818|
