### Ref papers
- [BEVDet: High-performance Multi-camera 3D Object Detection in Bird-Eye-View](https://arxiv.org/abs/2112.11790)

### Install
- clone [bevdet repo](https://github.com/HuangJunJie2017/BEVDet) and install it.
- link the projects folder to BEVDet folder. 
  - `cd /path/to/bevdet/`
  - `ln -s /path/to/sparsebit/examples/quantization_aware_training/nuscenes/bevdet/projects/ .`
- replace the tools folder in BEVDet with our tools
  - `cd /path/to/bevdet`
  - `mv ./tools tools.bak`
  - `ln -s /path/to/sparsebit/examples/quantization_aware_training/nuscenes/bevdet/tools/ .`

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

experiment | bits | mAP | NDS | Log
--- | --- | --- | --- | --- |
bevdet-r50 | float | 29.9 | 37.7 | - | 
bevdet-r50 | 8w8f | 29.8 | 38.1 | [google](https://drive.google.com/file/d/191UYMFEWaj03cK0TBe_vVnGkoUFKtyg0/view?usp=share_link) |
bevdet-r50 | 4w4f | 29.5 | 37.8 | [google](https://drive.google.com/file/d/19dkWpkPsSqgAvlHeevELmR15NhtQvKBU/view?usp=share_link) |
bevdet-vov99 | float | 36.3 | 43.6 | [google](https://drive.google.com/file/d/1cEpwJYNkLJaQkvsqqugBzyiypofPF-eQ/view?usp=share_link) |
bevdet-vov99 | 4w4f | 36.0 | 43.4  | [google](https://drive.google.com/file/d/1mIK_xWZHY9XWUasRSqOtz3X_exX20roP/view?usp=share_link) | 
