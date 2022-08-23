## Training
```
python3 -m torch.distributed.launch --nproc_per_node=4 --use_env main.py qconfig.yaml [imagenet-folder with train and val folders] -a [an architecture in torchvision.models] --multiprocessing-distributed -b 256 --lr 0.01 --print-freq 100 --pretrained
```

## How to get the dataset?
- first: `download ImageNet from https://image-net.org/download-images`
- second: `run a script from https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh`
