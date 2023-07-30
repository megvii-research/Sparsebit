## Command
```
python3 main.py --model_name yolov5n --data_path /PATH/TO/COCO
python3 main.py --model_name yolov5s --data_path /PATH/TO/COCO
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
  ```

## Calibration
-  Random sample image paths for calibration:


## Pretraind model

- Download float model:
    - [yolov5n]()
    - [yolov5s]()

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
|-----|-----|-----|-----|
|YOLOv5n|float|
|YOLOv5n|8w8f|27.2%|45.2%|58.1%|42.6%|
|
|YOLOv5s|float|
|YOLOv5s|8w8f|36.8%|56.4%|66.8%|52.0%|