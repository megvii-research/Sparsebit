import math
import torch
import torch.nn as nn
from torch.distributed.pipeline.sync import Pipe
import torch.distributed.pipeline.sync.utils as pipe_utils
from typing import Optional
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    _make_causal_mask,
    _expand_mask,
)


def get_first_device(model):
    if model.devices:
        return model.devices[0]
    else:
        return torch.cuda.current_device()


class LlamaEmbedLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )

    def _prepare_decoder_attention_mask(
        self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(self, input_ids, attention_mask):
        batch_size, seq_length = input_ids.shape
        past_key_values_length = 0
        # build position_ids
        position_ids = torch.arange(
            past_key_values_length,
            seq_length + past_key_values_length,
            dtype=torch.long,
            device=input_ids.device,
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        inputs_embeds = self.embed_tokens(input_ids)

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )

        return inputs_embeds, attention_mask, position_ids


class LlamaDecoderLayerWrapped(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask,
        position_ids,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, attention_mask, position_ids


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, attention_mask, position_ids):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class LlamaSequential(nn.Sequential):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        # self.padding_idx = config.pad_token_id
        # self.vocab_size = config.vocab_size
        # embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        embed_layer = LlamaEmbedLayer(config)
        d_layers = nn.ModuleList(
            [LlamaDecoderLayerWrapped(config) for _ in range(config.num_hidden_layers)]
        )
        norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        # self.post_init()

        layers = [embed_layer]
        layers.extend([layer for layer in d_layers])
        layers.extend([norm, lm_head])

        super().__init__(*layers)


class LlamaForCausalLMWrappedForPipe(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.model = LlamaSequential(self.config)

    def pipe_sequential_model(self, chunks=1, pp_checkpoint="never"):
        assert chunks >= 1 and isinstance(
            chunks, int
        ), "chunks must be a interger greater than 1"
        num_devices = torch.cuda.device_count()
        num_layers = len(self.model)

        layers_per_device = math.ceil(num_layers / num_devices)
        balance = [layers_per_device for _ in range(num_devices - 1)]
        last_device_layers = num_layers - layers_per_device * (num_devices - 1)
        if last_device_layers == 0:
            last_device_layers = 1
            balance[-1] -= 1
        balance.append(last_device_layers)

        model = pipe_utils.partition_model(self.model, balance)
        model_p = Pipe(model, chunks=chunks, checkpoint=pp_checkpoint)
        del model
        self.model = model_p

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        *args,
        **kwargs,
    ):
        try:
            first_device = get_first_device(self.model)
            input_ids = input_ids.to(first_device)
            attention_mask = attention_mask.to(first_device)
            outputs = self.model(input_ids, attention_mask).local_value()
        except Exception as e:
            raise RuntimeError(
                f"training failed on {torch.distributed.get_rank()}"
            ) from e
        hidden_states = outputs
        # hidden_states = self.norm(hidden_states)
        # logits = self.lm_head(hidden_states)
        return {"logits": hidden_states}
