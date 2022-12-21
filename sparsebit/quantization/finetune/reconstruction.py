import torch
from torch.fx import GraphModule, Node
from torch import fx, nn
from torch.nn import Module
import numpy as np
from sparsebit.quantization.modules import QuantOpr, QConv2d, QLinear
from enum import Enum


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)

def lp_loss(pred, tgt, p=2.0):
    """
    loss function measured in L_p Norm
    """
    return (pred - tgt).abs().pow(p).sum(1).mean()


def to_device(data, device="cpu"):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        for key in data:
            data[key] = to_device(data[key], device)
        return data
    elif isinstance(data, list):
        for idx, _ in enumerate(data):
            data[idx] = to_device(data[idx], device)
        return data
    else:
        return data


def tensor_detach(data):
    if isinstance(data, torch.Tensor):
        return data.detach()
    elif isinstance(data, dict):
        for key in data:
            data[key] = tensor_detach(data[key])
        return data
    elif isinstance(data, list):
        data = [tensor_detach(dat) for dat in data]
    else:
        return data


def set_quant_state(subgraph, w_quant=False, a_quant=False):
    for n, m in subgraph.named_modules():
        if isinstance(m, QuantOpr):
            m.set_quant(w_quant, a_quant)


def set_adaround_soft_targets(subgraph, soft_targets: bool):
    for name, layer in subgraph.named_modules():
        if isinstance(layer, ADAROUND_SUPPORT_OPR):
            weight_quantizer = layer.weight_quantizer
            weight_quantizer.soft_targets = soft_targets


def save_inp_oup_data(
    model: GraphModule,
    inp_module: Module,
    oup_module: Module,
    cali_data: list,
    store_inp=True,
    store_oup=True,
    keep_gpu: bool = True,
):
    """
    Save input data and output data of a particular layer/block over calibration dataset.
    :param cali_data: calibration data set
    :param keep_gpu: put saved data on GPU for faster optimization
    :return: input and output data
    """
    device = next(model.parameters()).device
    if store_inp:
        assert inp_module is not None
        inp_saver = DataSaverHook(
            store_input=store_inp, store_output=False, stop_forward=(not store_oup)
        )
        inp_handle = inp_module.register_forward_hook(inp_saver)
    if store_oup:
        assert oup_module is not None
        oup_saver = DataSaverHook(
            store_input=False, store_output=store_oup, stop_forward=True
        )
        oup_handle = oup_module.register_forward_hook(oup_saver)
    cached = ([], [])
    with torch.no_grad():
        for batch in cali_data:
            try:
                _ = model(to_device(batch, device))
            except StopForwardException:
                pass
            if store_inp:
                if keep_gpu:
                    cached[0].append(
                        [tensor_detach(inp) for inp in inp_saver.input_store]
                    )
                else:
                    cached[0].append(
                        [
                            to_device(tensor_detach(inp), "cpu")
                            for inp in inp_saver.input_store
                        ]
                    )  # tuple/list one
            if store_oup:
                if keep_gpu:
                    cached[1].append(tensor_detach(oup_saver.output_store))
                else:
                    cached[1].append(
                        to_device(tensor_detach(oup_saver.output_store), "cpu")
                    )
    if store_inp:
        inp_handle.remove()
    if store_oup:
        oup_handle.remove()
    torch.cuda.empty_cache()
    return cached


def subgraph_reconstruction(subgraph, cached_inps, cached_oups, config):
    a_para = []
    for name, layer in subgraph.named_modules():
        if isinstance(layer, QuantOpr) and hasattr(layer, "input_quantizer") and layer.input_quantizer.TYPE == "Quadapter":
            print("learn the input_quantizer alpha for {}".format(name))
            a_para += [layer.input_quantizer.alpha]
    assert len(a_para) != 0, "must specify Quadapter before ptq finetuning!"
    MAX_COUNT = config.A.QUANTIZER.TRAINING.ITERS
    a_opt = torch.optim.Adam(a_para, lr=config.A.QUANTIZER.TRAINING.LR)
    a_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        a_opt, T_max=MAX_COUNT, eta_min=0.0
    )

    set_quant_state(subgraph, w_quant=True, a_quant=True)
    reconstruct_feature(
        subgraph,
        cached_inps,
        cached_oups,
        lp_loss,
        a_opt,
        a_scheduler,
        config.A.QUANTIZER.TRAINING.ITERS,
    )


def reconstruct_feature(
    subgraph, cached_inps, cached_oups, loss_func, a_opt, a_scheduler, iters
):
    device = next(subgraph.parameters()).device
    sz = len(cached_inps[0])
    num_args = len(cached_inps)
    error_meter = AverageMeter("Error", ":6.2f", Summary.AVERAGE)
    for i in range(iters):
        idx = np.random.randint(0, sz)
        cur_args = []
        for a in range(num_args):
            cur_inp = to_device(cached_inps[a][idx], device)
            cur_args.append(cur_inp)
        cur_args = tuple(cur_args)
        cur_out = to_device(cached_oups[idx], device)
        a_opt.zero_grad()
        out_quant = subgraph(*cur_args)
        err = loss_func(out_quant, cur_out)
        error_meter.update(err, cur_inp.size(0))
        if i%500 == 0:
            print("error at iter {}: {}".format(i, error_meter.avg))
        err.backward()
        a_opt.step()
        if a_scheduler:
            a_scheduler.step()
    torch.cuda.empty_cache()



class StopForwardException(Exception):
    """
    Used to throw and catch an exception to stop traversing the graph
    """

    pass


class DataSaverHook:
    """
    Forward hook that stores the input and output of a layer/block
    """

    def __init__(self, store_input=False, store_output=False, stop_forward=False):
        self.store_input = store_input
        self.store_output = store_output
        self.stop_forward = stop_forward

        self.input_store = None
        self.output_store = None

    def __call__(self, module, input_batch, output_batch):
        if self.store_input:
            self.input_store = input_batch
        if self.store_output:
            self.output_store = output_batch
        if self.stop_forward:
            raise StopForwardException
