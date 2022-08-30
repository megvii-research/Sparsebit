## Command
```
python3 main.py --model-path yolov3.darknet53.coco.pth --dataset-root ./coco   --qconfig qconfig.yaml
```

## Dataset
- Download coco2017 in https://cocodataset.org/.

- Extract all of these tars into one directory, which should have the following basic structure:

  ```
  coco
  └── train2017
  └── val2017
  └── annotations
  ```

## Pretraind model

- Download float model from https://drive.google.com/file/d/1I9J5ftpJRjUfIS1A3mZ5vxNxY4CCgVTS/view?usp=sharing

## Requirements
```
pycocotools==2.0.4
Pillow==8.4.0
```

## COCO Benchmark
- Task: COCO
- Eval data num: 5k
- Calibration data num: 16
- Weight bit: 8
- Feature bit: 8
- Weight observer: MinMax
- Feature observers: MinMax
- Backend: TensorRT (symmetric feature quantization)

|Model|Float|MinMax|MSE|
|-----|-----|-----|-----|
|YOLOV3-relu|59.6%|58.5%|58.8%|