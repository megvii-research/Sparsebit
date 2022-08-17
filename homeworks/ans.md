# Solution For Homework
## Q1:
- 量化之后各个模型应当有不同程度的精度下降;
- Resnet18精度下降极小，vgg16_bn次之，mobilenetv2精度下降较大
- mobilenetv2量化之后掉点严重的主要原因在于该网络中的深度可分离卷积，DFQ [<sup>1</sup>](#refer-anchor)中的数据显示，Depthwise Conv不同输出通道的动态范围差异较大，因此采用per-tensor的量化方式将会引入较大的量化误差，从而导致精度损失严重，采用per-channel的量化能够缓解精度损失的问题。



## Q2:
- 实验结果: 不同calibration size的推理精度损失均不高
### 原因分析：
- 预处理中，图像的Normalization操作可确保数据满足N(0,1)的高斯分布，而随机高斯噪声输入同样满足高斯分布要求，二者的分布、范围接近，因此精度损失不高;
- 噪声输入随机性较大，不如图片本身具有一定的特点，所以校准后推理精度略有下降。


## Q3:
### Code:
- NOTE：注意到data中存在Batch维，因此不能直接对max_val/min_val做tensor-wise的moving average，应当采用循环的写法遍历data中的所有图片.
```python
@register_observer
class Observer(BaseObserver):
    TYPE = "movingaverage"

    def __init__(self, config, qdesc):
        super(Observer, self).__init__(config, qdesc)
        # 1. 从config中获取ALPHA参数并打印
        self.alpha = config.OBSERVER.ALPHA
        print(f"A moving-average observer with alpha = {self.alpha} is used!")

    def calc_minmax(self):
        assert (
            len(self.data_cache) > 0
        ), "Before calculating the quant params, the observation of data should be done"
        # 2. Reshape data into [CAL_SIZE,C,HW]
        # e.g. 如果有256张calibration图像，每张size是[3,224,224]，则这里 data.shape  == [256,3,50176]
        if self.qdesc.ch_axis > 0:
            data = torch.cat(self.data_cache, axis=0)
            data = (
                data
                .reshape(data.shape[0], data.shape[self.qdesc.ch_axis], -1)
                .detach()
                .data
            )
        else:
            data = torch.cat(self.data_cache, axis=1)
            data = data.reshape(data.shape[self.qdesc.ch_axis], -1).detach().data

        self.reset_data_cache()
        # 3. 根据 per_channel/per_tensor 计算min/max_value per sample
        # if per_channel, min/max_value.shape == [CAL_SIZE,C]
        # if per_tensor, min/max_value shape == [CAL_SIZE]
        if self.is_perchannel:
            max_val = data.max(axis=2).values
            min_val = data.min(axis=2).values
        else:
            min_val, max_val = data.min(axis =1).values.min(axis = 1).values, data.max(axis=1).values.max(axis=1).values
        
        min_val = min_val.to(self.device)
        max_val = max_val.to(self.device)

        # 4. 迭代计算滑动平均值
        self.min_val = min_val[0]
        self.max_val = max_val[0]
        for i in range(1, len(min_val)):
            self.min_val = self.min_val * ( 1 - self.alpha) + min_val[i] * self.alpha
            self.max_val = self.max_val * (1 - self.alpha) + max_val[i] * self.alpha
            
        return self.min_val, self.max_val
```
- 实验结果: 
1. 不同Alpha的点数均小于Minmax observer
2. Alpha较大时，掉点更多

## Q4:

- Int8相对于FP16的主要优势在于，相同带宽下，Int8传输的数据量是FP16的两倍，而对于mul-add计算来说，二者在GPU中的速度并没有太大区别
- GPU是一个以高吞吐量为特点的设备，BatchSize较小时，其传输带宽并没有被完全利用，因此Int8和FP16的Throughput接近;当BatchSize较大时，Int8的优势能够体现出来，即传输数据的时间是FP16的一半，访存时间更少，因此Int8的Throughput高于Fp16，存在着近似于2倍的关系。


## Q5:
- 实验结果：
1. 权重的精度下降，最终推理精度也会随之下降
2. 二者Quant Acc1均在90%左右

<div id = "refer-anchor"></div>

## References
[1]  [Data-Free Quantization Throught Weight Equalization and Bias Correction](https://arxiv.org/pdf/1906.04721.pdf)