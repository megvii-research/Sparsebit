import torch


def _parse_config(cfg_file, default_cfg=None):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = default_cfg.clone()  # default config
    config.defrost()
    config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    config.merge_from_file(cfg_file)
    config.freeze()
    return config


def update_config(config, key, value):
    config.defrost()
    keys = key.split(".")

    def _set_config_attr(cfg, keys, value):
        if len(keys) > 1:
            cfg = getattr(cfg, keys[0].upper())
            _set_config_attr(cfg, keys[1:], value)
        else:
            setattr(cfg, keys[0].upper(), value)

    _set_config_attr(config, keys, value)
    config.freeze()
    return config
