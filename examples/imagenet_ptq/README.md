## Command
```
python3 main.py qconfig.yaml /data/public-datasets/imagenet/ -a resnet18 --pretrained
```

## How to get the dataset?
- first: `download ImageNet from https://image-net.org/download-images`
- second: `run a script from https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh`

## ImageNet Benchmark
- Task: ImageNet
- Eval data num: 50k
- Calib data num: 256
- Weight bit: 8
- Feature bit: 8
- Weight observer: MinMax
- Feature observers: As shown in following table
- Backend: TensorRT (symmetric feature quantization)

|Model|Float|MinMax|MSE|Percentile w/ alpha=1e-3|Moving average w/ ema_ratio=0.9|ACIQ|KL histogram|
|-----|-----|-----|-----|-----|-----|-----|-----|
|ResNet18|69.76%|69.544%|69.59%|68.42%|69.558%|69.544%|69.354%|
|ResNet50|76.146%|75.888%|76.042%|75.34%|76.026%|76.014%|75.634%|
|MobileNetV2|70.49%|68.468%|69.498%|69.242%|69.708%|68.896%|68.578%|
|EfficientNet-Lite0|75.394%|75.116%|75.148%|74.862%| 75.09%|75.116%|31.082%|
|RegNetX-600MF|75.034%|74.56%|74.776%|72.946%|74.692%|74.602%|74.458%|