# Copyright (c) 2024 The HuggingFace Inc. team.
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2025-05-20.
#
# Original file was released under Apache-2.0, with the full license text
# available at https://github.com/huggingface/transformers/blob/main/LICENSE.
#
# This modified file is released under the same license.
import collections.abc

import torch
from torch import nn
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

from transformers.activations import ACT2FN
from transformers.pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer

from ..dinov2_with_registers.configuration_dinov2_with_registers import Dinov2WithRegistersConfig
from ..dinov2_with_registers.modeling_dinov2_with_registers import (
    Dinov2WithRegistersSelfAttention,
    Dinov2WithRegistersPreTrainedModel,
    Dinov2WithRegistersEmbeddings,
    Dinov2WithRegistersPatchEmbeddings,
)

try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None



class Dinov2WithRegistersSelfAttention2(Dinov2WithRegistersSelfAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.IntTensor,
        max_seqlen: int,
        **kwargs,
    ) -> torch.Tensor:
        # print('total_q_len, hidden_states', hidden_states.size())

        total_q_len, _ = hidden_states.size()

        query_states = self.query(hidden_states)
        key_states = self.key(hidden_states)
        value_states = self.value(hidden_states)

        query_states = query_states.view(total_q_len, self.num_attention_heads, self.attention_head_size)
        key_states = key_states.view(total_q_len, self.num_attention_heads, self.attention_head_size)
        value_states = value_states.view(total_q_len, self.num_attention_heads, self.attention_head_size)

        context_layer = flash_attn_varlen_func(
            query_states.to(torch.bfloat16),
            key_states.to(torch.bfloat16),
            value_states.to(torch.bfloat16),
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=False,
        )
        # new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        outputs = context_layer.reshape(total_q_len, -1)
        return outputs

class Dinov2WithRegistersSelfOutput(nn.Module):
    """
    The residual connection is defined in Dinov2WithRegistersLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: Dinov2WithRegistersConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states

class Dinov2WithRegistersAttention(nn.Module):
    def __init__(self, config: Dinov2WithRegistersConfig) -> None:
        super().__init__()
        self.attention = Dinov2WithRegistersSelfAttention2(config)
        self.output = Dinov2WithRegistersSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads: Set[int]) -> None:
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.IntTensor,
        max_seqlen: int,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_outputs = self.attention(hidden_states, cu_seqlens, max_seqlen, **kwargs)

        attention_output = self.output(self_outputs, hidden_states)
        #edits, note here is only outputs
        # outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        outputs = attention_output
        return outputs

class Dinov2WithRegistersLayerScale(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.lambda1 = nn.Parameter(config.layerscale_value * torch.ones(config.hidden_size))

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return hidden_state * self.lambda1


def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output


class Dinov2WithRegistersDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)
    
class Dinov2WithRegistersMLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        in_features = out_features = config.hidden_size
        hidden_features = int(config.hidden_size * config.mlp_ratio)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        if isinstance(config.hidden_act, str):
            self.activation = ACT2FN[config.hidden_act]
        else:
            self.activation = config.hidden_act
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.fc1(hidden_state)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.fc2(hidden_state)
        return hidden_state


class Dinov2WithRegistersSwiGLUFFN(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        in_features = out_features = config.hidden_size
        hidden_features = int(config.hidden_size * config.mlp_ratio)
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8

        self.weights_in = nn.Linear(in_features, 2 * hidden_features, bias=True)
        self.weights_out = nn.Linear(hidden_features, out_features, bias=True)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.weights_in(hidden_state)
        x1, x2 = hidden_state.chunk(2, dim=-1)
        hidden = nn.functional.silu(x1) * x2
        return self.weights_out(hidden)

class Dinov2WithRegistersLayer(nn.Module):
    """This corresponds to the Block class in the original implementation."""

    def __init__(self, config: Dinov2WithRegistersConfig) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = Dinov2WithRegistersAttention(config)
        self.layer_scale1 = Dinov2WithRegistersLayerScale(config)
        self.drop_path = (
            Dinov2WithRegistersDropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()
        )

        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        if config.use_swiglu_ffn:
            self.mlp = Dinov2WithRegistersSwiGLUFFN(config)
        else:
            self.mlp = Dinov2WithRegistersMLP(config)
        self.layer_scale2 = Dinov2WithRegistersLayerScale(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.IntTensor,
        max_seqlen: int,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_attention_outputs = self.attention(
            self.norm1(hidden_states),  # in Dinov2WithRegisters, layernorm is applied before self-attention
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            # head_mask,
            # output_attentions=output_attentions,
        )
        # attention_output = self_attention_outputs[0]
        attention_output = self_attention_outputs
        attention_output = self.layer_scale1(attention_output)
        # outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = self.drop_path(attention_output) + hidden_states

        # in Dinov2WithRegisters, layernorm is also applied after self-attention
        layer_output = self.norm2(hidden_states)
        layer_output = self.mlp(layer_output)
        layer_output = self.layer_scale2(layer_output)

        # second residual connection
        layer_output = self.drop_path(layer_output) + hidden_states

        # outputs = (layer_output,) + outputs
        outputs = layer_output
        return outputs

class Dinov2WithRegistersEncoder(nn.Module):
    def __init__(self, config: Dinov2WithRegistersConfig):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([Dinov2WithRegistersLayer(config) for _ in range(config.num_hidden_layers)])
        # self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.IntTensor,
        max_seqlen: int,
        # head_mask: Optional[torch.Tensor] = None,
        # output_attentions: bool = False,
        # output_hidden_states: bool = False,
        # return_dict: bool = True,
    ):
        # all_hidden_states = () if output_hidden_states else None
        # all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            layer_outputs = layer_module(hidden_states, cu_seqlens, max_seqlen)
            hidden_states = layer_outputs

        return hidden_states

class Dinov2WithRegistersModel(Dinov2WithRegistersPreTrainedModel):
    def __init__(self, config: Dinov2WithRegistersConfig):
        super().__init__(config)
        self.config = config

        self.embeddings = Dinov2WithRegistersEmbeddings(config)
        self.encoder = Dinov2WithRegistersEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> Dinov2WithRegistersPatchEmbeddings:
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        packed_pixel_values: torch.Tensor,
        # packed_flattened_position_ids: torch.LongTensor,
        cu_seqlens: torch.IntTensor,
        max_seqlen: int,
        # pixel_values: Optional[torch.Tensor] = None,
        # bool_masked_pos: Optional[torch.Tensor] = None,
        # head_mask: Optional[torch.Tensor] = None,
        # output_attentions: Optional[bool] = None,
        # output_hidden_states: Optional[bool] = None,
        # return_dict: Optional[bool] = None,
    ) -> torch.Tensor:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, sequence_length)`):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0). Only relevant for
            pre-training.
        """
        # output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # output_hidden_states = (
        #     output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        # )
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # if pixel_values is None:
        #     raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        # head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(packed_pixel_values)

        #consider reshape here: embedding_output 
        B, S, D = embedding_output.size() # reshape to linear patch 
        embedding_output = embedding_output.reshape(B*S, D)

        encoder_outputs = self.encoder(
            embedding_output,
            cu_seqlens,
            max_seqlen,
            # head_mask=head_mask,
            # output_attentions=output_attentions,
            # output_hidden_states=output_hidden_states,
            # return_dict=return_dict,
        )
        sequence_output = encoder_outputs
        sequence_output = self.layernorm(sequence_output)

        sequence_output = sequence_output.reshape(B, S, D)
        patch_tokens = sequence_output[:, 1 + self.config.num_register_tokens :]

        return patch_tokens
        # pooled_output = sequence_output[:, 0, :]

        # if not return_dict:
        #     head_outputs = (sequence_output, pooled_output)
        #     return head_outputs + encoder_outputs[1:]

        # return BaseModelOutputWithPooling(
        #     last_hidden_state=sequence_output,
        #     pooler_output=pooled_output,
        #     hidden_states=encoder_outputs.hidden_states,
        #     attentions=encoder_outputs.attentions,
        # )
