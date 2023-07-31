## Command
```
python3 main.py --model_name yolov5n --qconfig_path qconfig.yaml --data_path /PATH/TO/COCO --checkpoint_path checkpoints/yolov5n.pth
python3 main.py --model_name yolov5s --qconfig_path qconfig.yaml --data_path /PATH/TO/COCO --checkpoint_path checkpoints/yolov5s.pth
```

## Dataset
-  Download and prepare coco2017 as described in YOLOv5 repo, which should have the following basic structure:

  ```
  coco
  └── images
    └── train2017
    └── val2017
  └── annotations
  └── labels
  └── train2017.txt
  └── val2017.txt
  ```

## Calibration
-  Random sample image paths for calibration:

  ```
  python3 random_sample_calib.py --data_path /PATH/TO/COCO
  ```

## Pretraind model
- create checkpoints dir:
  ```
  mkdir ./checkpoints
  ```
- Download float checkpoints:
    - [yolov5n](https://drive.google.com/file/d/1pcsVQHoHCZ4N0ZB8E2QfDFzCmKfSCOjz/view?usp=sharing)
    - [yolov5s](https://drive.google.com/file/d/1fsDtQtnmNfMM6n0CpslzTMca7xkiaWhq/view?usp=sharing)

## Requirements
```
pip install -r yolov5/requirements.txt
```

## COCO Benchmark
- Task: COCO
- Eval data num: 5k
- Calibration data num: 128
- Weight bit: 8
- Feature bit: 8
- Weight
  - Granularity: channel-wise
  - Scheme: symmetric
  - Observer: MinMax
- Feature
  - Granularity: tensor-wise
  - Scheme: asymmetric
  - Observer: MinMax

|Model|qconfig|mAP50-95|mAP50|prec|recall|
|-----|-----|-----|-----|-----|-----|
|YOLOv5n|float|27.7%|45.6%|57.5%|43.2%|
|YOLOv5n|8w8f|27.3%|45.2%|58.0%|42.8%|
||
|YOLOv5s|float|37.1%|56.6%|66.8%|52.1%|
|YOLOv5s|8w8f|36.7%|56.5%|66.2%|52.1%|