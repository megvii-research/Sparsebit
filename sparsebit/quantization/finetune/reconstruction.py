import torch
from torch.fx import GraphModule, Node
from torch import fx, nn
from torch.nn import Module
import numpy as np
from sparsebit.quantization.modules import QuantOpr, QConv2d, QLinear


ADAROUND_SUPPORT_OPR = (QConv2d, QLinear)


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
    w_para, a_para = [], []
    w_opt, w_scheduler = None, None
    for name, layer in subgraph.named_modules():
        if isinstance(layer, ADAROUND_SUPPORT_OPR):
            weight_quantizer = layer.weight_quantizer
            print("learn the weight_quantizer alpha for {}".format(name))
            w_para += [weight_quantizer.alpha]
        if isinstance(layer, QuantOpr) and not layer.fake_fused:
            if config.W.QUANTIZER.ADAROUND.GRANULARITY == "blockwise":
                input_quantizer = layer.input_quantizer
                if input_quantizer is not None:
                    print("learn the input_quantizer scale for {}".format(name))
                    a_para += [input_quantizer.scale]
    if len(a_para) != 0:
        if (
            config.A.QUANTIZER.TRAINING.ENABLE
            and config.W.QUANTIZER.TRAINING.ORDER != "together"
        ):
            MAX_COUNT = config.A.QUANTIZER.TRAINING.ITERS
        else:
            MAX_COUNT = config.W.QUANTIZER.TRAINING.ITERS
        a_opt = torch.optim.Adam(a_para, lr=config.A.QUANTIZER.TRAINING.LR)
        a_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            a_opt, T_max=MAX_COUNT, eta_min=0.0
        )
    else:
        a_opt, a_scheduler = None, None
    w_opt = torch.optim.Adam(w_para)

    loss_func = LossFunction(
        subgraph=subgraph,
        weight=config.W.QUANTIZER.ADAROUND.LAMBDA,
        max_count=config.W.QUANTIZER.TRAINING.ITERS,
        b_range=config.W.QUANTIZER.ADAROUND.B_RANGE,
        warm_up=config.W.QUANTIZER.ADAROUND.WARM_UP,
    )
    if config.A.QUANTIZER.TRAINING.ENABLE:
        loss_func_act = LossFunction(
            subgraph=subgraph,
            max_count=config.A.QUANTIZER.TRAINING.ITERS,
            disable_round_loss=True,
        )
    if (
        config.W.QUANTIZER.ADAROUND.GRANULARITY == "blockwise"
        and config.A.QUANTIZER.TRAINING.ENABLE
    ):
        assert a_opt is not None
        if config.W.QUANTIZER.TRAINING.ORDER == "before":
            print(
                "start training, we will train the weight_quantizer alpha first, then train the input_quantizer scale"
            )
            set_quant_state(subgraph, w_quant=True, a_quant=False)
            reconstruct_weight(
                subgraph,
                cached_inps,
                cached_oups,
                loss_func,
                w_opt,
                w_scheduler,
                config.W.QUANTIZER.TRAINING.ITERS,
            )
            set_quant_state(subgraph, w_quant=True, a_quant=True)
            reconstruct_feature(
                subgraph,
                cached_inps,
                cached_oups,
                loss_func_act,
                a_opt,
                a_scheduler,
                config.A.QUANTIZER.TRAINING.ITERS,
            )
        elif config.W.QUANTIZER.TRAINING.ORDER == "together":
            print(
                "start training, we will train the weight_quantizer alpha and the input_quantizer scale together"
            )
            set_quant_state(subgraph, w_quant=True, a_quant=True)
            reconstruct_together(
                subgraph,
                cached_inps,
                cached_oups,
                loss_func,
                w_opt,
                a_opt,
                w_scheduler,
                a_scheduler,
                config.W.QUANTIZER.TRAINING.ITERS,
            )
        elif config.W.QUANTIZER.TRAINING.ORDER == "after":
            print(
                "start training, we will train the input_quantizer scale first, then train weight_quantizer alpha"
            )
            set_quant_state(subgraph, w_quant=False, a_quant=True)
            reconstruct_feature(
                subgraph,
                cached_inps,
                cached_oups,
                loss_func_act,
                a_opt,
                a_scheduler,
                config.A.QUANTIZER.TRAINING.ITERS,
            )
            set_quant_state(subgraph, w_quant=True, a_quant=True)
            reconstruct_weight(
                subgraph,
                cached_inps,
                cached_oups,
                loss_func,
                w_opt,
                w_scheduler,
                config.W.QUANTIZER.TRAINING.ITERS,
            )
        else:
            raise "unsupport training order {}".format(
                config.W.QUANTIZER.TRAINING.ORDER
            )
    else:
        print("start training, we will only train the weight_quantizer alpha")
        set_quant_state(subgraph, w_quant=True, a_quant=True)
        reconstruct_weight(
            subgraph,
            cached_inps,
            cached_oups,
            loss_func,
            w_opt,
            w_scheduler,
            config.W.QUANTIZER.TRAINING.ITERS,
        )


def reconstruct_weight(
    subgraph, cached_inps, cached_oups, loss_func, w_opt, w_scheduler, iters
):
    device = next(subgraph.parameters()).device
    set_adaround_soft_targets(subgraph, True)
    sz = len(cached_inps[0])
    num_args = len(cached_inps)
    for i in range(iters):
        idx = np.random.randint(0, sz)
        cur_args = []
        for a in range(num_args):
            cur_inp = to_device(cached_inps[a][idx], device)
            cur_args.append(cur_inp)
        cur_args = tuple(cur_args)
        cur_out = to_device(cached_oups[idx], device)
        w_opt.zero_grad()
        out_quant = subgraph(*cur_args)
        err = loss_func(out_quant, cur_out)
        err.backward()
        w_opt.step()
        if w_scheduler:
            w_scheduler.step()
    torch.cuda.empty_cache()
    set_adaround_soft_targets(subgraph, False)


def reconstruct_feature(
    subgraph, cached_inps, cached_oups, loss_func, a_opt, a_scheduler, iters
):
    device = next(subgraph.parameters()).device
    sz = len(cached_inps[0])
    num_args = len(cached_inps)
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
        err.backward()
        a_opt.step()
        if a_scheduler:
            a_scheduler.step()
    torch.cuda.empty_cache()


def reconstruct_together(
    subgraph,
    cached_inps,
    cached_oups,
    loss_func,
    w_opt,
    a_opt,
    w_scheduler,
    a_scheduler,
    iters,
):
    device = next(subgraph.parameters()).device
    set_adaround_soft_targets(subgraph, True)
    sz = len(cached_inps[0])
    num_args = len(cached_inps)
    for i in range(iters):
        idx = np.random.randint(0, sz)
        cur_args = []
        for a in range(num_args):
            cur_inp = to_device(cached_inps[a][idx], device)
            cur_args.append(cur_inp)
        cur_args = tuple(cur_args)
        cur_out = to_device(cached_oups[idx], device)
        if a_opt:
            a_opt.zero_grad()
        w_opt.zero_grad()
        out_quant = subgraph(*cur_args)
        err = loss_func(out_quant, cur_out)
        err.backward()
        w_opt.step()
        if a_opt:
            a_opt.step()
        if w_scheduler:
            w_scheduler.step()
        if a_scheduler:
            a_scheduler.step()
    torch.cuda.empty_cache()
    set_adaround_soft_targets(subgraph, False)


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


class LinearTempDecay:
    def __init__(self, t_max=10000, warm_up=0.2, start_b=20, end_b=2):
        self.t_max = t_max
        self.start_decay = warm_up * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        if t < self.start_decay:
            return self.start_b
        elif t > self.t_max:
            return self.end_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))


class LossFunction:
    r"""loss function to calculate mse reconstruction loss and relaxation loss
    use some tempdecay to balance the two losses.
    """

    def __init__(
        self,
        subgraph: Module,
        weight: float = 1.0,
        max_count: int = 10000,
        b_range: tuple = (20, 2),
        warm_up: float = 0.0,
        p: float = 2.0,
        disable_round_loss: bool = False,
    ):

        self.subgraph = subgraph
        self.weight = weight
        self.loss_start = max_count * warm_up
        self.p = p
        self.disable_round_loss = disable_round_loss

        self.temp_decay = LinearTempDecay(
            max_count, warm_up=warm_up, start_b=b_range[0], end_b=b_range[1]
        )
        self.count = 0

    def __call__(self, pred, tgt):
        """
        Compute the total loss for adaptive rounding:
        rec_loss is the quadratic output reconstruction loss, round_loss is
        a regularization term to optimize the rounding policy
        :param pred: output from quantized model
        :param tgt: output from FP model
        :return: total loss function
        """
        self.count += 1
        rec_loss = lp_loss(pred, tgt, p=self.p)

        b = self.temp_decay(self.count)
        if self.count < self.loss_start or self.disable_round_loss:
            round_loss = 0
        else:
            round_loss = 0
            for layer in self.subgraph.modules():
                if isinstance(layer, ADAROUND_SUPPORT_OPR):
                    round_vals = layer.weight_quantizer.get_soft_targets()
                    round_loss += (
                        self.weight * (1 - ((round_vals - 0.5).abs() * 2).pow(b)).sum()
                    )

        total_loss = rec_loss + round_loss
        if self.count % 500 == 0:
            print(
                "Total loss:\t{:.3f} (rec:{:.3f}, round:{:.3f})\tb={:.2f}\tcount={}".format(
                    float(total_loss), float(rec_loss), float(round_loss), b, self.count
                )
            )
        return total_loss