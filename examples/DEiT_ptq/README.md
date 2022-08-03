# DEiT example

## preparation

The `DEiT` pretrained model is the checkpoint from https://github.com/facebookresearch/deit . The example will automatically download the checkpoint using `torch.hub.load`.

The datasets used in example are calibration dataset and validation dataset.

- For calibration dataset, follow the statements in https://github.com/megvii-research/Sparsebit/blob/homeworks/homeworks/quant_homework.md#resource and download `imagenet-1k dataset`. Use `./calibration_data/calibration/` as the path to calibration dataset.
  ```shell
  tar -xvf imagenet-1k-images.tar -C path_to_calibration_data/calibration
  ln -s path_to_calibration_data calibration_data
  ```

- For validation dataset, download from https://image-net.org/ and move validation images to labeled subfolders, using [this script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh). Use `./validation/` as the path to validation dataset.
  ```shell
  tar -xvf ILSVRC2012_img_val.tar -C path_to_validation_data
  cd path_to_validation_data
  wget https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
  chmod +x valprep.sh
  ./valprep.sh
  cd path_to_example
  ln -s path_to_validation_data validation_data
  ```

The datasets are loaded with `torchvision.datasets.ImageFolder`. Using custom datasets for calibration and validation is OK.

## Usage

```shell
python3 main.py qconfig.yaml
```
Use argument `-b batch_size` to assign batch_size if the default batch_size(=128) is too large.

## Metrics

|model|float32 acc|8w8f acc|
|-|-|-|
|DEiT-tiny|72.026|70.778|
|DEiT-base|81.742|81.152|
