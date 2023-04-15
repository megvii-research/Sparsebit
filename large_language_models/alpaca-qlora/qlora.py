import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import QuantLinear
from peft import PeftModel, PeftConfig, PeftModelForCausalLM, LoraModel
from peft.mapping import _prepare_lora_config, _prepare_prompt_learning_config
from peft.utils import PromptLearningConfig, _set_trainable, transpose
from peft.tuners.lora import LoraConfig, LoraLayer
from transformers.utils import PushToHubMixin
from qmatmul import Quant4Matmul


class LoraQModel(LoraModel):
    """
    Creates Low Rank Adapter (Lora) model from a pretrained transformers model.

    Args:
        model ([`transformers.PreTrainedModel`]): The model to be adapted.
        config ([`LoraConfig`]): The configuration of the Lora model.

    Returns:
        `torch.nn.Module`: The Lora model.

    Example::

        >>> from transformers import AutoModelForSeq2SeqLM, LoraConfig >>> from peft import LoraModel, LoraConfig >>>
        config = LoraConfig(
            peft_type="LORA", task_type="SEQ_2_SEQ_LM", r=8, lora_alpha=32, target_modules=["q", "v"],
            lora_dropout=0.01, )
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base") >>> lora_model = LoraModel(config, model)

    **Attributes**:
        - **model** ([`transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`LoraConfig`]): The configuration of the Lora model.
    """

    def _find_and_replace(self):
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        is_target_modules_in_base_model = False
        is_hf_device_map_available = hasattr(self.model, "hf_device_map")
        kwargs = {
            "r": self.peft_config.r,
            "lora_alpha": self.peft_config.lora_alpha,
            "lora_dropout": self.peft_config.lora_dropout,
            "fan_in_fan_out": self.peft_config.fan_in_fan_out,
            "merge_weights": (
                self.peft_config.merge_weights or self.peft_config.inference_mode
            )
            and not is_hf_device_map_available,
        }
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if isinstance(self.peft_config.target_modules, str):
                target_module_found = re.fullmatch(self.peft_config.target_modules, key)
            else:
                target_module_found = any(
                    key.endswith(target_key)
                    for target_key in self.peft_config.target_modules
                )
            if target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = self._get_submodules(key)
                bias = target.bias is not None
                if (
                    isinstance(target, torch.nn.Linear)
                    and self.peft_config.enable_lora is None
                ):
                    new_module = Linear(
                        target.in_features, target.out_features, bias=bias, **kwargs
                    )
                elif (
                    isinstance(target, QuantLinear)
                    and self.peft_config.enable_lora is None
                ):
                    new_module = QLinear(
                        target.in_features, target.out_features, target.bit, **kwargs
                    )
                elif self.peft_config.enable_lora is not None:
                    kwargs.update({"enable_lora": self.peft_config.enable_lora})
                    if isinstance(target, Conv1D):
                        in_features, out_features = (
                            target.weight.ds_shape
                            if hasattr(target.weight, "ds_shape")
                            else target.weight.shape
                        )
                    else:
                        in_features, out_features = (
                            target.in_features,
                            target.out_features,
                        )
                        if kwargs["fan_in_fan_out"]:
                            warnings.warn(
                                "fan_in_fan_out is set to True but the target module is not a Conv1D. "
                                "Setting fan_in_fan_out to False."
                            )
                            kwargs[
                                "fan_in_fan_out"
                            ] = self.peft_config.fan_in_fan_out = False
                    new_module = MergedLinear(
                        in_features, out_features, bias=bias, **kwargs
                    )
                self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {self.peft_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.qweight = old_module.qweight
        new_module.scales = old_module.scales
        new_module.zeros = old_module.zeros
        if old_module.bias is not None:
            new_module.bias = old_module.bias
        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "lora_" in name:
                module.to(old_module.qweight.device)

        del old_module


class PeftQModel(PeftModel, PushToHubMixin, torch.nn.Module):
    """
    Parameter-Efficient Fine-Tuning Model. Base model encompassing various Peft methods.

    Args:
        model ([`PreTrainedModel`]): The base transformer model used for Peft.
        peft_config ([`PeftConfig`]): The configuration of the Peft model.


    **Attributes**:
        - **base_model** ([`PreTrainedModel`]) -- The base transformer model used for Peft.
        - **peft_config** ([`PeftConfig`]) -- The configuration of the Peft model.
        - **modules_to_save** (`list` of `str`) -- The list of sub-module names to save when
        saving the model.
        - **prompt_encoder** ([`PromptEncoder`]) -- The prompt encoder used for Peft if
        `isinstance(self.peft_config, PromptLearningConfig)`.
        - **prompt_tokens** (`torch.Tensor`) -- The virtual prompt tokens used for Peft if
        `isinstance(self.peft_config, PromptLearningConfig)`.
        - **transformer_backbone_name** (`str`) -- The name of the transformer
        backbone in the base model if `isinstance(self.peft_config, PromptLearningConfig)`.
        - **word_embeddings** (`torch.nn.Embedding`) -- The word embeddings of the transformer backbone
        in the base model if `isinstance(self.peft_config, PromptLearningConfig)`.
    """

    def __init__(self, model, peft_config: PeftConfig):
        PushToHubMixin.__init__(self)
        torch.nn.Module.__init__(self)
        self.peft_config = peft_config
        self.base_model = model
        self.config = self.base_model.config
        self.modules_to_save = None
        if isinstance(self.peft_config, PromptLearningConfig):
            self._setup_prompt_encoder()
        else:
            self.base_model = LoraQModel(peft_config, model)
        if getattr(self.peft_config, "modules_to_save", None) is not None:
            self.modules_to_save = self.peft_config.modules_to_save
            _set_trainable(self)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PeftQModelForCausalLM(PeftModelForCausalLM, PeftQModel):
    """
    Peft Qmodel for Causal LM

    Args:
        model ([`PreTrainedModel`]): Base transformer model
        peft_config ([`PeftConfig`]): Peft config.


    Example::

        >>> from transformers import AutoModelForCausalLM >>> from peft import PeftModelForCausalLM, get_peft_config
        >>> config = {
                'peft_type': 'PREFIX_TUNING', 'task_type': 'CAUSAL_LM', 'inference_mode': False, 'num_virtual_tokens':
                20, 'token_dim': 1280, 'num_transformer_submodules': 1, 'num_attention_heads': 20, 'num_layers': 36,
                'encoder_hidden_size': 1280, 'prefix_projection': False, 'postprocess_past_key_value_function': None
            }
        >>> peft_config = get_peft_config(config) >>> model = AutoModelForCausalLM.from_pretrained("gpt2-large") >>>
        peft_model = PeftModelForCausalLM(model, peft_config) >>> peft_model.print_trainable_parameters() trainable
        params: 1843200 || all params: 775873280 || trainable%: 0.23756456724479544
    """

    def __init__(self, model, peft_config: PeftConfig):
        PeftQModel.__init__(self, model, peft_config)
        self.base_model_prepare_inputs_for_generation = (
            self.base_model.prepare_inputs_for_generation
        )


def get_peft_qmodel(model, peft_config):
    """
    Returns a Peft qmodel object from a model and a config.

    Args:
        model ([`transformers.PreTrainedModel`]): Model to be wrapped.
        peft_config ([`PeftConfig`]): Configuration object containing the parameters of the Peft model.
    """

    model_config = model.config.to_dict()
    peft_config.base_model_name_or_path = model.__dict__.get("name_or_path", None)
    if not isinstance(peft_config, PromptLearningConfig):
        peft_config = _prepare_lora_config(peft_config, model_config)
    else:
        peft_config = _prepare_prompt_learning_config(peft_config, model_config)
    assert (
        peft_config.task_type == "QUANT_CAUSAL_LM"
    ), "only support quant causal language model"
    return PeftQModelForCausalLM(model, peft_config)


class QLinear(QuantLinear, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bit: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs,
    ):
        QuantLinear.__init__(self, in_features, out_features, bit, **kwargs)
        LoraLayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=False,
        )

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Linear(in_features, r, bias=False)
            self.lora_B = nn.Linear(r, out_features, bias=False)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.qweight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.zeros_(self.lora_A.weight)  # a walkaround for weight is nan
            nn.init.zeros_(self.lora_B.weight)
            self.lora_A.weight.data = nn.init.kaiming_uniform(
                self.lora_A.weight, a=math.sqrt(5)
            )
            if (
                torch.isnan(self.lora_A.weight).sum() > 0
                or self.lora_A.weight.sum() == 0
            ):
                print("debug nan")
                from IPython import embed

                embed()

    def forward(self, x: torch.Tensor):
        if self.disable_adapters:
            if self.r > 0 and self.merged:
                self.weight.data -= (
                    transpose(
                        self.lora_B.weight @ self.lora_A.weight, self.fan_in_fan_out
                    )
                    * self.scaling
                )
                self.merged = False

            return F.linear(
                x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias
            )
        elif self.r > 0 and not self.merged:
            result = super().forward(x)
            if self.r > 0:
                expected_dtype = result.dtype
                if x.dtype != torch.float32:
                    x = x.float()
                output = (
                    self.lora_B(self.lora_A(self.lora_dropout(x))).to(expected_dtype)
                    * self.scaling
                )
                result += output
            return result
        else:
            return F.linear(
                x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias
            )
