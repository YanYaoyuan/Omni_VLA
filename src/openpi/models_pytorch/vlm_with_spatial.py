from typing import Literal

import pytest
import torch
from torch import nn
from transformers import GemmaForCausalLM
from transformers import PaliGemmaForConditionalGeneration
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma import modeling_gemma

from safetensors.torch import load_file

from openpi.vggt.models.vggt import VGGT
from openpi.vlm_expert.dinov2_with_registers.modeling_dinov2_with_registers import Dinov2WithRegistersModel
from openpi.vlm_expert.dinov2_with_registers.modular_dinov2_with_registers import Dinov2WithRegistersConfig

import os

MODEL_PATH = "/data/base_model/pi0_base_torch/model.safetensors"

VGGT_PRETRAINED_PATH = "/data/base_model/VGGT-1B/model.safetensors"

# Define the complete layer computation function for gradient checkpointing
def compute_layer_complete(
    layer_idx, inputs_embeds, attention_mask, position_ids, reasoning_expert, spatial_expert, action_expert
):
    models = [reasoning_expert.language_model, spatial_expert, action_expert]
    query_states = []
    key_states = []
    value_states = []

    # for i, hidden_states in enumerate(inputs_embeds):
    #     layer = models[i].model.layers[layer_idx]
    #     hidden_states = layer.input_layernorm(hidden_states)  # noqa: PLW2901
    #     input_shape = hidden_states.shape[:-1]
    #     hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
    #     if layer.self_attn.q_proj.weight.dtype == torch.bfloat16:
    #         hidden_states = hidden_states.to(dtype=torch.bfloat16)
    #     #query_state = layer.self_attn.q_norm(layer.self_attn.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    #     query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    #     key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    #     value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    #     # key_state = layer.self_attn.k_norm(layer.self_attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    #     # value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    #     query_states.append(query_state)
    #     key_states.append(key_state)
    #     value_states.append(value_state)
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].model.layers[layer_idx]
        hidden_states = layer.input_layernorm(hidden_states)  # noqa: PLW2901

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        query_states.append(query_state)
        key_states.append(key_state)
        value_states.append(value_state)
    # Concatenate and process attention
    query_states = torch.cat(query_states, dim=2)
    key_states = torch.cat(key_states, dim=2)
    value_states = torch.cat(value_states, dim=2)

    seq_len = query_states.shape[2]

    # 如何生成1650长度？？？
    max_rotary_len = 1635
    query_states = query_states[:, :, :max_rotary_len, :]
    key_states   = key_states[:, :, :max_rotary_len, :]
    value_states = value_states[:, :, :max_rotary_len, :]
    seq_len = query_states.shape[2]

    dummy_tensor = torch.zeros(
        query_states.shape[0],
        seq_len,
        query_states.shape[-1],
        device=query_states.device,
        dtype=query_states.dtype,
    )


    cos, sin = reasoning_expert.language_model.model.rotary_emb(dummy_tensor, position_ids)
    # cos, sin = reasoning_expert.get_rotary_embedding(seq_len)

    # ⚠ 确保长度对齐 Q/K
    cos = cos[:, :seq_len, :]
    sin = sin[:, :seq_len, :]
    
    query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
        query_states, key_states, cos, sin, unsqueeze_dim=1
    )
    batch_size = query_states.shape[0]
    scaling = reasoning_expert.language_model.model.layers[layer_idx].self_attn.scaling
    # Attention computation
    att_output, _ = modeling_gemma.eager_attention_forward(
        reasoning_expert.language_model.model.layers[layer_idx].self_attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling,
    )
    # Get head_dim from the current layer, not from the model
    head_dim = reasoning_expert.language_model.model.layers[layer_idx].self_attn.head_dim
    num_attention_heads = reasoning_expert.language_model.model.layers[layer_idx].self_attn.config.num_attention_heads
    att_output = att_output.reshape(batch_size, -1, 1 * num_attention_heads * head_dim)
    # Process layer outputs
    outputs_embeds = []
    start_pos = 0
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].model.layers[layer_idx]
        end_pos = start_pos + hidden_states.shape[1]
        if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
            att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
        out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])
        # first residual
        residual_len = out_emb.shape[1]  # 当前 att_output 对应长度
        hidden_states = hidden_states[:, :residual_len, :]
        out_emb = out_emb + hidden_states
        after_first_residual = out_emb.clone()
        out_emb = layer.post_attention_layernorm(out_emb)
        # Convert to bfloat16 if the next layer (mlp) uses bfloat16
        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
            out_emb = out_emb.to(dtype=torch.bfloat16)
        out_emb = layer.mlp(out_emb)
        # second residual
        out_emb = out_emb + after_first_residual
        outputs_embeds.append(out_emb)
        start_pos = end_pos
    return outputs_embeds


class VLMWithSpatialActionExpertModel(
    nn.Module
):
    "VLM model with spatial expert with acthion expert"
    def __init__(
        self,
        reasoning_expert_config,
        spatial_expert_config,
        action_expert_config,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
    ):
        super().__init__()

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = reasoning_expert_config.width
        vlm_config_hf.text_config.intermediate_size = reasoning_expert_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = reasoning_expert_config.num_heads
        vlm_config_hf.text_config.head_dim = reasoning_expert_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = reasoning_expert_config.depth
        vlm_config_hf.text_config.num_key_value_heads = reasoning_expert_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        # vlm_config_hf.text_config.adarms_cond_dim = reasoning_expert_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"

        self.reasoning_expert = PaliGemmaForConditionalGeneration(config=vlm_config_hf)

        # 2. 加载本地权重
        state_dict = load_file(MODEL_PATH, device = "cpu")

        # 全量搜索所有 key 中是否包含 VLM
        vlm_keys = [k for k in state_dict.keys() if "paligemma_with_expert.paligemma" in k]

        print(f"Found {len(vlm_keys)} keys for reasoning_expert:")
        for k in vlm_keys:
            print(k)

        paligemma_state = {
            k.replace("paligemma_with_expert.paligemma.", "", 1): v
            for k, v in state_dict.items()
            if k.startswith("paligemma_with_expert.paligemma.")
        }

        gemma_state = {
            k.replace("paligemma_with_expert.gemma_expert.", "", 1): v
            for k, v in state_dict.items()
            if k.startswith("paligemma_with_expert.gemma_expert.")
        }


        missing_keys, unexpected_keys = self.reasoning_expert.load_state_dict(
            paligemma_state,
            strict=False    # 或 False，见下面说明
        )

        print(f"Loaded reasoning_expert: {len(paligemma_state)} params")
        print("Missing keys (first 20):", missing_keys[:20])
        print("Unexpected keys (first 20):", unexpected_keys[:20])


        "Spatial expert config"
        spatial_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=spatial_expert_config.head_dim,
            hidden_size=spatial_expert_config.width,
            intermediate_size=spatial_expert_config.mlp_dim,
            num_attention_heads=spatial_expert_config.num_heads,
            num_hidden_layers=spatial_expert_config.depth,
            num_key_value_heads=spatial_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            # use_adarms=use_adarms[1],
            # adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )
        

        self.vggt_encoder = VGGT(enable_camera=False,
                                    enable_point=False,
                                    enable_depth=False,
                                    enable_track=False,
                                    feature_only=True,
                                )
        state_dict_vggt = load_file(VGGT_PRETRAINED_PATH, device="cpu")
        missing, unexpected = self.vggt_encoder.load_state_dict(
            state_dict_vggt,
            strict=False
        )

        print("VGGT missing:", missing)
        print("VGGT unexpected:", unexpected)
        # self.spatial_projector = nn.Linear(768, spatial_expert_config_hf.width)
        self.spatial_expert = GemmaForCausalLM(config=spatial_expert_config_hf)

        "Actor expert config"
        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            # use_adarms=use_adarms[1],
            # adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        
        self.action_expert = GemmaForCausalLM(config=action_expert_config_hf)
        self.action_expert.model.embed_tokens = None

        self.to_bfloat16_for_selected_params(precision)

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            # "vision_tower.vision_model.embeddings.patch_embedding.weight",
            # "vision_tower.vision_model.embeddings.patch_embedding.bias",
            # "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)


    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | pytest.Cache | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        if adarms_cond is None:
            adarms_cond = [None, None]
        if inputs_embeds[1] is None and inputs_embeds[2] is None:
            prefix_output = self.reasoning_expert.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_hidden_states=True,
            )
            past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.hidden_states[-1]
            middle_output = None
            suffix_output = None
            
        elif inputs_embeds[0] is None and inputs_embeds[2] is None:
            middle_output = self.spatial_expert.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_hidden_states=True,
            )
            past_key_values = middle_output.past_key_values
            prefix_output = None
            middle_output = middle_output.hidden_states[-1]
            suffix_output = None
            
        elif inputs_embeds[0] is None and inputs_embeds[1] is None:
            suffix_output = self.action_expert.forward(
                inputs_embeds=inputs_embeds[2],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_hidden_states=True,
            )
            past_key_values = None
            prefix_output = None
            middle_output = None
            suffix_output = suffix_output.hidden_states[-1]
        else:
            models = [self.reasoning_expert.language_model, self.spatial_expert, self.action_expert]
            num_layers = self.reasoning_expert.config.text_config.num_hidden_layers

            # Check if gradient checkpointing is enabled for any of the models
            use_gradient_checkpointing = (
                hasattr(self.action_expert, "gradient_checkpointing")
                and self.spatial_expert.gradient_checkpointing
                and self.action_expert.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            # Process all layers with gradient checkpointing if enabled
            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        use_reentrant=False,
                        preserve_rng_state=False,
                        reasoning_expert=self.reasoning_expert,
                        spatial_expert=self.spatial_expert, 
                        action_expert=self.action_expert,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        reasoning_expert=self.reasoning_expert,
                        spatial_expert=self.spatial_expert, 
                        action_expert=self.action_expert,
                    )

            # final norm
            def compute_final_norms(inputs_embeds):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb = models[i].model.norm(hidden_states)
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            # Apply gradient checkpointing to final norm if enabled
            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms,
                    inputs_embeds,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds)

            past_key_values = None
            prefix_output = outputs_embeds[0]
            middle_output = outputs_embeds[1]
            suffix_output = outputs_embeds[2]

        return [prefix_output, middle_output, suffix_output], past_key_values