def mse_loss(pred, tgt, is_perchannel=False):
    if is_perchannel:
        return ((pred - tgt) ** 2).mean(-1)
    else:
        return ((pred - tgt) ** 2).mean()
