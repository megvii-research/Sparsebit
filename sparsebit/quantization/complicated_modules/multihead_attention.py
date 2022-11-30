import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from sparsebit.quantization.complicated_modules import register_complicated_module

@register_complicated_module(sources=[nn.MultiheadAttention])
class MultiheadAttention(nn.Module):
    """MultiheadAttention层。
    量化输入在build_quantizer中处理, 通过在输入上增加QIdentity层来解决。
    是MultipleModulesQuantOpr的子类。
    """

    def __init__(self, org_module=None):
        super().__init__()
        self.embed_dim=org_module.embed_dim
        self.num_heads=org_module.num_heads
        self.batch_first=org_module.batch_first
        self._qkv_same_embed_dim=org_module._qkv_same_embed_dim
        self.in_proj_weight=org_module.in_proj_weight
        self.bias_k=org_module.bias_k
        self.bias_v=org_module.bias_v
        self.in_proj_bias=org_module.in_proj_bias
        self.add_zero_attn=org_module.add_zero_attn
        self.out_proj=org_module.out_proj
        self.q_proj_weight=org_module.q_proj_weight
        self.k_proj_weight=org_module.k_proj_weight
        self.v_proj_weight=org_module.v_proj_weight
        self.dropout = org_module.dropout

        self.q_in_proj = nn.Linear(org_module.embed_dim, org_module.embed_dim)
        self.q_in_proj.weight.data = org_module.in_proj_weight[:org_module.embed_dim]
        self.q_in_proj.bias.data = org_module.in_proj_bias[:org_module.embed_dim]
        self.k_in_proj = nn.Linear(org_module.embed_dim, org_module.embed_dim)
        self.k_in_proj.weight.data = org_module.in_proj_weight[org_module.embed_dim:2*org_module.embed_dim]
        self.k_in_proj.bias.data = org_module.in_proj_bias[org_module.embed_dim:2*org_module.embed_dim]
        self.v_in_proj = nn.Linear(org_module.embed_dim, org_module.embed_dim)
        self.v_in_proj.weight.data = org_module.in_proj_weight[2*org_module.embed_dim:]
        self.v_in_proj.bias.data = org_module.in_proj_bias[2*org_module.embed_dim:]
        self.out_proj = org_module.out_proj

    def forward(self, query, key, value, key_padding_mask = None,
                need_weights = True, attn_mask = None):
        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]
        
        _, b, _ = query.shape

        query_proj = self.q_in_proj(query).reshape(-1, b*self.num_heads, self.embed_dim//self.num_heads).permute(1,0,2)
        key_proj = self.k_in_proj(key).reshape(-1, b*self.num_heads, self.embed_dim//self.num_heads).permute(1,2,0)/math.sqrt(self.embed_dim//self.num_heads)
        value_proj = self.v_in_proj(value).reshape(-1, b*self.num_heads, self.embed_dim//self.num_heads).permute(1,0,2)
        qk = torch.matmul(query_proj, key_proj)
        if key_padding_mask is not None:
            qk = qk.masked_fill(key_padding_mask.repeat(self.num_heads, 1).unsqueeze(1), float('-inf'))
        qk = torch.softmax(qk, dim=-1)
        qkv = torch.matmul(qk, value_proj).permute(1,0,2).reshape(-1,b,self.embed_dim)
        output = self.out_proj(qkv)


        # if not self._qkv_same_embed_dim:
        #     attn_output, attn_output_weights = F.multi_head_attention_forward(
        #         query, key, value, self.embed_dim, self.num_heads,
        #         self.in_proj_weight, self.in_proj_bias,
        #         self.bias_k, self.bias_v, self.add_zero_attn,
        #         self.dropout, self.out_proj.weight, self.out_proj.bias,
        #         training=self.training,
        #         key_padding_mask=key_padding_mask, need_weights=need_weights,
        #         attn_mask=attn_mask, use_separate_proj_weight=True,
        #         q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
        #         v_proj_weight=self.v_proj_weight)
        # else:
        #     attn_output, attn_output_weights = F.multi_head_attention_forward(
        #         query, key, value, self.embed_dim, self.num_heads,
        #         self.in_proj_weight, self.in_proj_bias,
        #         self.bias_k, self.bias_v, self.add_zero_attn,
        #         self.dropout, self.out_proj.weight, self.out_proj.bias,
        #         training=self.training,
        #         key_padding_mask=key_padding_mask, need_weights=need_weights,
        #         attn_mask=attn_mask)

        return output, None