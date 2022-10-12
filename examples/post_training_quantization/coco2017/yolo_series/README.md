## Command
```
python3 main.py --arch yolov3 --model-path yolov3.darknet53.coco.pth --dataset-root ./coco --qconfig qconfig.yaml
python3 main.py --arch yolov4 --model-path yolov4.coco.pth --dataset-root ./coco --qconfig qconfig.yaml --wo-input-norm
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

- Download float model:
    - [yolov3.darknet53.coco.pth](https://drive.google.com/file/d/1I9J5ftpJRjUfIS1A3mZ5vxNxY4CCgVTS/view?usp=sharing)
    - [yolov4.coco.pth](https://drive.google.com/file/d/1863jh81hfnVqBVPstuEalEMGex0fN4Mm/view?usp=sharing)

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
|YOLOV3|59.6%|58.5%|58.8%|
|YOLOV4|74.0%|71.0%|73.3%|