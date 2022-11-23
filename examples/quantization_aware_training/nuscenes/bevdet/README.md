### Install
- clone [bevdet repo](https://github.com/HuangJunJie2017/BEVDet) and install it.
- link the project folder to BEVDet folder. `ln -s ./projects /path/to/bevdet/`
- replace the tools folder in BEVDet with our tools

### Training
```
./tools/dist_qat_train.sh /path/to/config /path/to/qconfig /path/to/pretrained-float-model #gpu
```
- For example:
```
./tools/dist_qat_train.sh ./projects/configs/bevdet-r50-g4b8-e12.py ./projects/configs/qconfigs/qconfig_r50_lsq_4w4f.yaml ./checkpoints/bevdet-r50.pth 4
```

### Evaluation
```
python3 ./tools/qat_test.py /path/to/config /path/to/qconfig /path/to/checkpoints --eval=bbox
```
- For example
```
python3 ./tools/qat_test.py  ./projects/configs/bevdet-r50-g4b8-e12.py ./projects/configs/qconfigs/qconfig_r50_lsq_4w4f.yaml ./checkpoints/bevdet-r50-4w4f.pth --eval=bbox
```

### Results
- experiments setting:
  - in 4w4f, we set the wbit & abit of first conv and head-conv to 8

experiment | bits | mAP | NDS | Model | Log
--- | --- | --- | --- | --- | --- |
bevdet-r50 | float | 0.299 | 0.377 | - | - | 
bevdet-r50 | 8w8f | 0.298 | 0.381 | [google]() | [google]() |
bevdet-r50 | 4w4f | 0.295 | 0.378 | [google]() | [google]() |
bevdet-vov99 | float | 0.363 | 0.436 | [google]() | [google]() |
bevdet-vov99 | 4w4f | 0.360 | 0.434  | [google]() | [google]() | 
