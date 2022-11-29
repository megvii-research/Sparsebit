import torch


def check_torch_version(target_version):
    version_wo_cuda = torch.__version__.split("+")[0]  # del cuda info
    int_curr_version = [int(x) for x in version_wo_cuda.split(".")]
    int_tgt_version = [int(x) for x in target_version.split(".")]
    for cv, tv in zip(int_curr_version, int_tgt_version):
        if cv > tv:
            return True
        if cv == tv:
            continue
        if cv < tv:
            return False
    return True  # equal
