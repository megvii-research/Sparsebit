import torch
import numpy as np
import copy
from functools import partial
from scipy import stats
from typing import List, Tuple
from sparsebit.quantization.observers import Observer as BaseObserver
from sparsebit.quantization.observers import register_observer
from sparsebit.quantization.common import Granularity


torch.multiprocessing.set_sharing_strategy("file_system")


def run_distributed(
    func,
    total_run_times: int,
    input_tuples: List[Tuple] = [tuple()],
    use_cpus: int = 4,
):
    """
    A wrapper for multiprocessing
    input:
        func -> parallel running function
        total_run_times: numbers to run
        input_tuples: list of tuple(inputs) with length==total_run_times
    output:
        out -> list of outputs with length==total_run_times
    """
    import multiprocessing as mp

    with mp.Pool(processes=use_cpus) as p:
        # use processes=None will degenerate to single process in rlaunch
        # but processes=None will be faster than processes=4 without rlaunch
        output = []
        for i in range(total_run_times):
            output.append(p.apply_async(func, input_tuples[i]))
        p.close()
        p.join()

    out = []
    for i in range(total_run_times):
        out.append(output[i].get())
    return out


def get_best_threshold(data, hist_min, hist_max, bit, bins):
    histogram = torch.histc(data, bins=bins, min=hist_min, max=hist_max)
    bin_width = (hist_max - hist_min) / bins
    dst_bins = 2**bit - 1
    new_th = calibrate_entropy(histogram.numpy(), bin_width, bins, dst_bins)
    return new_th


def calibrate_entropy(distribution, bin_width, src_bins, dst_bins=255):
    # num_quantized_bins=255, bins):
    zero_bin_idx = src_bins // 2
    num_half_quantized_bins = dst_bins // 2
    # thresholds = np.zeros([self.bins // 2 + 1 - num_quantized_bins // 2])
    divergence = np.zeros([src_bins // 2 + 1 - dst_bins // 2])
    for i in range(num_half_quantized_bins, zero_bin_idx):
        p_bin_idx_start = zero_bin_idx - i
        p_bin_idx_stop = zero_bin_idx + i + 1
        sliced_nd_hist = np.zeros([p_bin_idx_stop - p_bin_idx_start])
        p = copy.deepcopy(distribution[p_bin_idx_start:p_bin_idx_stop])
        p[0] += sum(distribution[:p_bin_idx_start])
        p[p_bin_idx_stop - p_bin_idx_start - 1] = sum(distribution[p_bin_idx_stop:])
        sliced_nd_hist = copy.deepcopy(distribution[p_bin_idx_start:p_bin_idx_stop])
        num_merged_bins = sliced_nd_hist.size // dst_bins
        quantized_bins = np.zeros([dst_bins])
        for j in range(dst_bins):
            start = j * num_merged_bins
            stop = start + num_merged_bins
            quantized_bins[j] = sliced_nd_hist[start:stop].sum()
        quantized_bins[-1] += sliced_nd_hist[dst_bins * num_merged_bins :].sum()
        is_nonzeros = (p != 0).astype(np.int64)
        q = np.zeros(sliced_nd_hist.size, dtype=np.float64)
        for j in range(dst_bins):
            start = j * num_merged_bins
            if j == dst_bins - 1:
                stop = -1
            else:
                stop = start + num_merged_bins
            norm = is_nonzeros[start:stop].sum()
            if norm != 0:
                q[start:stop] = float(quantized_bins[j]) / float(norm)
        q[p == 0] = 0
        p[p == 0] = 0.0001
        q[q == 0] = 0.0001
        divergence[i - dst_bins] = stats.entropy(p, q)
    min_kl_divergence = np.argmin(divergence)
    th = bin_width * min_kl_divergence
    return th


@register_observer
class Observer(BaseObserver):
    TYPE = "kl_histogram"

    def __init__(self, config, qdesc):
        super(Observer, self).__init__(config, qdesc)
        self.bins = 2048

    def calc_minmax(self, bit):
        if self.is_perchannel:
            data = self.data_cache.get_data_for_calibration(
                Granularity.CHANNELWISE
            ).cpu()
            channel = data.shape[0]
            abs_max = data.abs().max(axis=1).values
            _min = torch.empty(channel)
            _max = torch.empty(channel)
            func = partial(get_best_threshold, bit=bit, bins=self.bins)
            th = run_distributed(
                func,
                total_run_times=channel,
                use_cpus=24,
                input_tuples=[
                    tuple(
                        [
                            data[c],
                            -abs_max[c],
                            abs_max[c],
                        ]
                    )
                    for c in range(channel)
                ],
            )
            for c in range(channel):
                _min[c] = -th[c] if data[c].min() < 0 else 0
                _max[c] = th[c]
            self.max_val = _max.to(self.device)
            self.min_val = _min.to(self.device)
        else:
            data = self.data_cache.get_data_for_calibration(Granularity.LAYERWISE).cpu()
            abs_max = data.abs().max()
            th = get_best_threshold(
                data=data,
                hist_min=-abs_max,
                hist_max=abs_max,
                bit=bit,
                bins=self.bins,
            )
            self.max_val = th.to(self.device)
            self.min_val = (
                -th.to(self.device)
                if data.min() < 0
                else torch.zeros(1).to(self.device)
            )
        return self.min_val, self.max_val
