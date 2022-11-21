## Training
```
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --qconfig qconfig.yaml --model [the architecture you want to choose] --batch-size 128 --data-path [imagenet-folder with train and val folders] --output_dir [the path of log] --pretrained [the path of a float trained model] --lr 0.0001 --epochs 100
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

|model| float | DoReFa|PACT|LSQ|
|---|---|---|---|---|
| DeiT-tiny |72.2 | - | - | 71.27 |
| DeiT-Base | | - | - |  |
| Swin-tiny |81.148 | - | - | 80.818|

## Notes
- Following the quant setting of most papers, we only quantize all the weights and inputs involved in matrix multiplication, and do not quantize the softmax operation and layer normalization. 
- The base learning rate is 1e-4 not 5e-4, because we are fine-tuning from a pretrained float model
- To make torch.fx can trace a model from timm, we must comment the asserts in timm. [link](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/layers/patch_embed.py#L42-L43)
