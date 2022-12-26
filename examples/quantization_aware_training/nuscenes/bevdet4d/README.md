### Ref papers
- [BEVDet4D: Exploit Temporal Cues in Multi-camera 3D Object Detection](https://arxiv.org/abs/2203.17054)

### Install
Here we applied the third-party implementation of BEVDet4d from BEVDet repo as our baseline.
- clone [bevdet repo](https://github.com/HuangJunJie2017/BEVDet) and install it.
- link the projects folder to BEVDet folder. 
  - `cd /path/to/bevdet/`
  - `ln -s /path/to/Sparsebit/examples/quantization_aware_training/nuscenes/bevdet4d/projects/ .`
- replace the tools folder in BEVDet with our tools
  - `cd /path/to/bevdet`
  - `mv ./tools tools.bak`
  - `ln -s /path/to/Sparsebit/examples/quantization_aware_training/nuscenes/bevdet4d/tools/ .`

### Training
```
./tools/dist_qat_train.sh /path/to/config /path/to/qconfig /path/to/pretrained-float-model #gpu
```
- For example:
```
./tools/dist_qat_train.sh ./projects/configs/bevdet4d-r50-g7b4-e12.py ./projects/configs/qconfigs/qconfig_r50_lsq_4w4f.yaml ./checkpoints/bevdet4d-r50.pth 7
```

### Evaluation
```
python3 ./tools/qat_test.py /path/to/config /path/to/qconfig /path/to/checkpoints --eval=bbox
```
- For example
```
python3 ./tools/qat_test.py ./projects/configs/bevdet4d-r50-g7b4-e12.py   ./projects/configs/qconfigs/qconfig_r50_lsq_4w4f.yaml ./checkpoints/bevdet4d-r50-4w4f.pth --eval=bbox
```

### Results
- experiments setting:
  - in 4w4f, we set the wbit & abit of first conv and head-conv to 8

experiment | bits | mAP | NDS | Log
--- | --- | --- | --- | --- |
bevdet4d-r50 | float | 32.2 | 45.7 | - | 
bevdet4d-r50 | 4w4f | 31.7 | 45.8 | [google](https://drive.google.com/file/d/116-PIyVBFGe-dzeBpmNGJmNcHxIw0pke/view?usp=share_link) |
