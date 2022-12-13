### Ref papers
- [BEVDepth: Acquisition of Reliable Depth for Multi-view 3D Object Detection](https://arxiv.org/pdf/2206.10092.pdf)

### Install
Here we applied the third-party implementation of BEVDepth from BEVDet repo as our baseline.
- clone [bevdet repo](https://github.com/HuangJunJie2017/BEVDet) and install it.
- link the projects folder to BEVDet folder. 
  - `cd /path/to/bevdet/`
  - `ln -s /path/to/Sparsebit/examples/quantization_aware_training/nuscenes/bevdepth/projects/ .`
- replace the tools folder in BEVDet with our tools
  - `cd /path/to/bevdet`
  - `mv ./tools tools.bak`
  - `ln -s /path/to/Sparsebit/examples/quantization_aware_training/nuscenes/bevdepth/tools/ .`

### Training
```
./tools/dist_qat_train.sh /path/to/config /path/to/qconfig /path/to/pretrained-float-model #gpu
```
- For example:
```
./tools/dist_qat_train.sh ./projects/configs/bevdepth-r50-wo-dcnv2-g4b6-e12.py ./projects/configs/qconfigs/qconfig_r50_lsq_4w4f.yaml ./checkpoints/bevdepth-r50-wo-dcnv2.pth 4
```

### Evaluation
```
python3 ./tools/qat_test.py /path/to/config /path/to/qconfig /path/to/checkpoints --eval=bbox
```
- For example
```
python3 ./tools/qat_test.py ./projects/configs/bevdepth-r50-wo-dcnv2-g4b6-e12.py  ./projects/configs/qconfigs/qconfig_r50_lsq_4w4f.yaml ./checkpoints/bevdepth-r50-wo-dcnv2-4w4f.pth --eval=bbox
```

### Results
- experiments setting:
  - in 4w4f, we set the wbit & abit of first conv and head-conv to 8

experiment | bits | mAP | NDS | Log
--- | --- | --- | --- | --- |
bevdepth-r50-wo-dcnv2 | float | 33.6 | 40.9 | [google](https://drive.google.com/file/d/1dgVHVEfMv38lDHnzbVWuC6AMLM6hbCA6/view?usp=share_link) |
bevdepth-r50-wo-dcnv2 | 8w8f | 33.52 | 41.12 | [google](https://drive.google.com/file/d/1WkfI8RHb1Pt6NbmS8irJ6o1ulJSE_HrU/view?usp=share_link) |
bevdepth-r50-wo-dcnv2 | 4w4f | 33.27 | 40.72 | [google](https://drive.google.com/file/d/1Fy76UMwuRSijmPwMCyAz_eXkk3Qngm_u/view?usp=share_link) |
bevdepth-vov99-wo-dcnv2 | float | 37.84 | 44.05 | [google](https://drive.google.com/file/d/1g0qapk9XIfECfqb6iYXsC1Xx_gmwH2t1/view?usp=share_link) |
bevdepth-vov99-wo-dcnv2 | 4w4f | 38.34 | 44.68  | [google](https://drive.google.com/file/d/1HgVWKrVeHzKwTb-Ee3JcURa09vcRfvg8/view?usp=share_link) |
