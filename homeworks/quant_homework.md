# Homeworks
### Q1: 请模仿cifar10样例，构造imagenet工程并获取vgg16_bn/resnet18/mobilenetv2的PTQ实验结果

| Model | Calibration Size|Quant Config | Float Acc1 | Quant Acc1 |
| :-------- | :----------------- | :------  |:------------- | :---|
| Resnet18  | 256  |W8A8          |  | |
| vgg16_bn | 256  |W8A8          | ||
| mobilenetv2| 256 |W8A8      | ||

### Q2: 请基于resnet18实验, 把calibration-set里面的图片换成标准高斯噪声输入, 当calibration-set大小为1, 10, 100时, 请问精度分别是多少, 精度不是0或者很低的原因是什么呢?
| Model | Calibration Size | Quant Config | Acc1 |
| :------- | :------------         | :---------- |  :-------|
| ResNet18 | 1 | W8A8 |  |
| ResNet18 | 10 | W8A8 | |
| ResNet18 | 100 | W8A8 | |
- 原因分析: 

### Q3: 请增加moving-average observer，并重新运行题目一的Resnet18，观察实验结果。
*Hint:*
$$\text{value} = \text{value} * (1 - \alpha) + \text{new} *\alpha$$

| Model | Calibration Set|Quant Config |Alpha |Quant Acc1 |
| :-------- | :----------------- | :------  |:------------- |:--|
| Resnet18  | 256  |W8A8  | 0.5|      |
| Resnet18  | 256 | W8A8  | 0.9|      |
| Resnet18  | 256 | W8A8  | 0.99|    |
- 代码:
```python
# TO DO
```

### Q4: 请使用trtexec加载导出的ONNX模型，在batch=1,32,128,256情况下，相较于fp16的加速情况，请分析int8/fp16为什么在batch不同时会有显著差异？

| Model | Batch Size|Quant Config | Int8 QPS |Fp16 QPS | Hardware |
| :-------- | :----------------- | :------  |:------------- | :---|:-- |
| Resnet18  | 1 |W8A8          | | | |
| Resnet18 | 32 |W8A8          | | |  |
| Resnet18 | 128 |W8A8      |  | | |
| Resnet18 | 256 |W8A8      |  | | |

- 原因分析：

### Q5: 仿造cifar10 qat样例训练resnet18模型，要求bit分别为4w4f,2w4f，观察精度变化。

| Model |Quant Config| Quant Acc1 |
| :-------- | :----------------- | :------  |
| Resnet18  | 4w4f     |   |
| Resnet18  |2w4f      |   |

## Resource
- Resnet20 Cifar10预训练模型
  - Google Drive：https://drive.google.com/file/d/1XPsG8_vYEY1hx_S82eaQeNe2cDoUBNCc/view?usp=sharing
  - 百度网盘： https://pan.baidu.com/s/1JbuWeaLvECrYdGuyHlsd4A, 提取码: 9gp9
- Imagenet-1k数据集
  - Google Drive：https://drive.google.com/file/d/1YjPgiFP06vvH8s-QqRtZcBFxudmhlEuL/view?usp=sharing
  - 百度网盘：https://pan.baidu.com/s/1-W4vp-yIFQCzjcv6kGtJ5A, 提取码: c2sj
- Imagenet验证集
  - Imagenet官网 https://image-net.org/ 下载
  - 解压步骤可参考：https://blog.csdn.net/qq_45588019/article/details/125642466
  - linux环境下可通过以下命令下载
```bash
# 验证集 ILSVRC2012_img_val.tar
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar --no-check-certificate
# 标签映射文件 ILSVRC2012_devkit_t12.tar.gz
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz --no-check-certificate
```
