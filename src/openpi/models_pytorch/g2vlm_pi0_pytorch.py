
from __future__ import annotations

import math
from typing import List, Literal, Optional

from PIL import Image


from safetensors.torch import load_file
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma import modeling_gemma
from torch.nn.attention.flex_attention import create_block_mask

from openpi import models_pytorch as _mpp  # 官方
from openpi.models import gemma as _gemma  # 官方
from openpi.models_pytorch import preprocessing_pytorch as _pp  # 官方
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel  # 官方
from openpi.vlm_expert.dinov2_with_registers import Dinov2WithRegistersConfig
from openpi.vlm_expert.dinov2_with_registers import Dinov2WithRegistersModel
from openpi.vlm_expert.g2vlm import G2VLM
from openpi.vlm_expert.g2vlm import Dinov2WithRegistersConfig
from openpi.vlm_expert.g2vlm import Dinov2WithRegistersModel
from openpi.vlm_expert.g2vlm import G2VLMConfig
from openpi.vlm_expert.g2vlm import Qwen2VLConfig
from openpi.vlm_expert.g2vlm import Qwen2VLForCausalLM
from openpi.vlm_expert.g2vlm.qwen2vl import Qwen2VLForCausalLM
from openpi.vlm_expert.qwen2 import Qwen2Tokenizer
from openpi.vlm_expert.qwen2.configuration_qwen2 import Qwen2Config
from openpi.vlm_expert.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from openpi.vlm_expert.qwen2vl.configuration_qwen2_vl import Qwen2VLVisionConfig
from openpi.vlm_expert.qwen2vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel
from openpi.vlm_expert.qwen2vl.modeling_qwen2_vl import apply_multimodal_rotary_pos_emb
from openpi.vlm_expert.qwen2vl.modeling_qwen2_vl_vit import Qwen2VisionTransformerPretrainedModel

from ..data_vlm.data_utils import add_special_tokens, create_sparse_mask
from ..data_vlm.data_utils import pil_img2rgb
from ..data_vlm.transforms import ImageTransform
from ..data_vlm.transforms import InternVLImageTransform
from ..data_vlm.transforms import QwenVL2ImageTransform
from ..data_vlm.transforms_vggt import DinoImageNormalizeTransform
from ..data_vlm.transforms_vggt import DinoImageTransform

from ..data_vlm.data_utils import (
    create_sparse_mask, 
    get_flattened_position_ids_extrapolate, 
    get_flattened_position_ids_interpolate,
    get_rope_index_image_3D,
    get_rope_index_image_3D_dino,
    patchify, 
)

from openpi.vlm_expert.g2vlm.qwen2vl import Qwen2VLForCausalLM, Qwen2VLConfig, NaiveCache
import logging
import sys
from pathlib import Path
from typing import Literal

import os

# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def apply_mrope_to_expert(q, k, cos, sin):
    """
    针对 Qwen2-VL 的 M-RoPE 逻辑：将 head_dim 拆分为 T, H, W 三部分分别旋转
    q, k: [Batch, Heads, Seq, Dim]
    cos, sin: [3, Batch, Seq, Dim] (由 rope_module 生成)
    """
    # 1. 按照 Qwen2-VL 官方比例拆分 head_dim (1/2, 1/4, 1/4)
    dim = cos.shape[-1]
    m_cos = torch.cat([
        cos[0, ..., :dim//2],          # Temporal
        cos[1, ..., dim//2:3*dim//4],    # Height
        cos[2, ..., 3*dim//4:]           # Width
    ], dim=-1)
    
    m_sin = torch.cat([
        sin[0, ..., :dim//2],
        sin[1, ..., dim//2:3*dim//4],
        sin[2, ..., 3*dim//4:]
    ], dim=-1)

    # 2. 增加 Heads 维度用于广播: [B, 1, L, D]
    m_cos = m_cos.unsqueeze(1)
    m_sin = m_sin.unsqueeze(1)

    # 3. 执行旋转 (FP32 计算以保稳)
    q_out = (q.float() * m_cos) + (rotate_half(q.float()) * m_sin)
    k_out = (k.float() * m_cos) + (rotate_half(k.float()) * m_sin)
    
    return q_out.to(q.dtype), k_out.to(k.dtype)


# Add 20250110
def apply_rotary_pos_emb_vision_3d(q, k, cos, sin):
    """
    针对 Qwen2-VL M-RoPE 的 3D 旋转应用
    q, k: [Batch, Heads, Seq, Dim]
    cos, sin: [3, Batch, Seq, Dim] (由 rope_module 生成)
    """
    # 1. 维度对齐：将 cos/sin 插入 Heads 维度以便广播 [3, B, 1, L, D]
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    
    # 2. 核心逻辑：Qwen2-VL 将 head_dim 拆分为 T, H, W 三部分
    # 通常比例为: T(1/2), H(1/4), W(1/4)
    dim = q.shape[-1]
    
    # 构造混合旋转矩阵
    # 这种方式保证了 q 的不同通道分别吸收了不同轴的位置信息
    m_cos = torch.cat([
        cos[0, ..., :dim//2],          # 时间分量旋转前一半维度
        cos[1, ..., dim//2:3*dim//4],    # 高度分量旋转中间 1/4
        cos[2, ..., 3*dim//4:]           # 宽度分量旋转最后 1/4
    ], dim=-1)
    
    m_sin = torch.cat([
        sin[0, ..., :dim//2],
        sin[1, ..., dim//2:3*dim//4],
        sin[2, ..., 3*dim//4:]
    ], dim=-1)

    # 3. 执行旋转计算
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    # 提升精度计算防止 NaN
    orig_dtype = q.dtype
    q, k = q.float(), k.float()
    
    q_embed = (q * m_cos) + (rotate_half(q) * m_sin)
    k_embed = (k * m_cos) + (rotate_half(k) * m_sin)
    
    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)

def get_rope_index_for_hidden(attention_mask: torch.Tensor):
    """
    Returns position_ids of shape (batch, seq_len) compatible with rotary_emb
    """
    if attention_mask is None:
        raise ValueError("attention_mask must be provided")

    # cumsum 生成 position_ids
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    return position_ids  # (batch, seq_len)


def build_transform(pixel=224):
    image_transform = QwenVL2ImageTransform(pixel, pixel, 14)

    return image_transform

def load_model_and_tokenizer(model_path):
    llm_config = Qwen2VLConfig.from_json_file(os.path.join(model_path, "text_config.json"))

    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = 'Qwen2VLMoTDecoderLayer'  

    vit_config = Qwen2VLVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.patch_size =14

    dino_config = Dinov2WithRegistersConfig.from_json_file(os.path.join(model_path, "dino_config.json"))

    config = G2VLMConfig(
        visual_und=True,
        visual_recon=True, # Dino use
        llm_config=llm_config, 
        vit_config=vit_config,
        dino_config=dino_config,
        vit_max_num_patch_per_side=36,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    language_model = Qwen2VLForCausalLM(llm_config).to(device)
    vit_model      = Qwen2VisionTransformerPretrainedModel(vit_config).to(device)
    dino_model = Dinov2WithRegistersModel(dino_config).to(device)

    model = G2VLM(language_model, vit_model, dino_model, config)

    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    vit_image_transform = QwenVL2ImageTransform(768, 768, 14)
    dino_transform = DinoImageNormalizeTransform(target_size=518)

    model_state_dict_path = os.path.join(model_path, "model.safetensors")
    model_state_dict = load_file(model_state_dict_path, device="cpu")
    msg = model.load_state_dict(model_state_dict, strict=False)
    print(msg)
    del model_state_dict
    model = model.cuda().eval()

    return model, tokenizer, new_token_ids , vit_image_transform, dino_transform, llm_config

# ---------- 1. 三专家 MoT ----------
class G2VLMWithActorExpertModel(nn.Module):
    """
    官方 PaliGemmaWithExpertModel 的“三专家”版：
    - prefix:  image+text  →  Semantic  Expert (PaliGemma, 冻结)
    - prefix:  dino→3D     →  Geometric Expert (G2VLM, 冻结)
    - suffix:  state+action → Action    Expert (Gemma-300M, 可训)
    共享 Self-Attention，FFN 按 token 类型路由。
    """

    """G2VLM model with action expert for PI0, replacing PaliGemmaWithExpertModel."""

    def __init__(
        self,
        g2_vlm_path,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
        image_size: int = 224,
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()

        # If G2VLM model is provided, use it directly
        g2_model, tokenizer, new_token_ids , vit_image_transform, dino_transform, llm_config= load_model_and_tokenizer(g2_vlm_path)
        device = g2_model.device
        self.g2vlm = g2_model.to(device = device)
        self.vit_image_transform = build_transform()# set 224
        # self.vit_image_transform = self.vit_image_transform.to(device = device)
        self.visiontower = g2_model.vit_model.to(device = device)
        self.dino_transform = dino_transform
        self.dinoTower = g2_model.dino_model.to(device = device)
        self.dinoProjector = g2_model.dino2llm
        self.llm_config = llm_config


        # Create action expert (Gemma model) similar to PaliGemmaWithExpertModel
        from transformers.models.auto import CONFIG_MAPPING
        from transformers.models.gemma import GemmaForCausalLM

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=128,
            hidden_size=llm_config.hidden_size,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=llm_config.num_attention_heads, # need same as G@VLM
            num_hidden_layers= 28, # need = to g2vlm language
            num_key_value_heads=llm_config.num_key_value_heads,
            vocab_size=self.g2vlm.language_model.vocab_size,  # Match PaliGemma vocab size
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        self.action_expert = GemmaForCausalLM(config=action_expert_config_hf).to(device = device)
        self.action_expert.model.embed_tokens = None  # We'll use shared embeddings

        self.action_gate = nn.Parameter(torch.ones(28))
        
        # 存储当前 batch 的 grid 信息，用于构建 position_ids
        self.current_vit_grid = []
        self.current_dino_grid = []

        # action_expert_config_hf = Qwen2Config(
        #     hidden_size=llm_config.hidden_size,
        #     intermediate_size=action_expert_config.mlp_dim,
        #     num_hidden_layers=action_expert_config.depth,
        #     num_attention_heads = llm_config.num_attention_heads, # need same as G@VLM
        #     num_key_value_heads=llm_config.num_key_value_heads,
        #     vocab_size=self.g2vlm.language_model.vocab_size,
        #     torch_dtype=torch.float32,
        #     attention_dropout=0.0,
        # )

        # self.action_expert = Qwen2ForCausalLM(
        #     config=action_expert_config_hf
        # ).to(device)
        # self.action_expert.model.embed_tokens = None

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
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def _ensure_transforms(self):
        """Lazily create image transforms for VIT and DINO, following G2VLM's own pipeline."""
        if getattr(self, "_vit_transform", None) is None:
            if "QwenVL2ImageTransform" in globals() and QwenVL2ImageTransform is not None:
                # Use the same settings as g2vlm_utils.load_model_and_tokenizer
                self._vit_transform = QwenVL2ImageTransform(768, 768, 14)
            else:
                logging.warning("QwenVL2ImageTransform is not available; VIT embeddings will be disabled.")
                self._vit_transform = None

        if getattr(self, "_dino_transform", None) is None:
            if "DinoImageTransform" in globals() and DinoImageTransform is not None:
                self._dino_transform = DinoImageTransform(target_size=518)
            else:
                logging.warning("DinoImageTransform is not available; DINO embeddings will be disabled.")
                self._dino_transform = None


    def embed_image(self, image: torch.Tensor):
        """
        image: Tensor[B, C, H, W]  (raw RGB, 0~1 or 0~255，取决于 transform)
        return:
            {
                "semantic":  Tensor[B, N_vit, D],
                "geometric": Tensor[B, N_dino, D],
                "vit_grid":  Tensor[1, 3],  # [T, H, W]
                "dino_grid": Tensor[1, 3],  # [T, H, W]
            }

        Add 20250110: 返回 grid grid_thw 信息
        """

        # --- 🚀 核心诊断：检查输入像素 ---
        print("-" * 30)
        print(f"DEBUG [Image Raw]: dtype: {image.dtype}")
        print(f"DEBUG [Image Raw]: shape: {image.shape}")
        print(f"DEBUG [Image Raw]: min: {image.min().item():.4f}")
        print(f"DEBUG [Image Raw]: max: {image.max().item():.4f}")
        print(f"DEBUG [Image Raw]: mean: {image.mean().item():.4f}")
        print("-" * 30)

        # --- 🚀 保命锁 1：防止全黑图片导致 NaN ---
        # 如果图像所有值都一样（方差为0），给它加一点点极其微小的噪声
        # 无条件加一个极小的扰动 (1e-6 几乎不影响训练，但能防止全平图像)
        # 或者直接把 allclose 移到 Gradient Checkpoint 之外
        image = image + torch.randn_like(image) * 1e-6

        device = image.device
        B, C, H, W = image.shape

        # 如果只有一张图
        if image.dim() == 3:                      # [3, H, W]
            image = image.unsqueeze(0)            # [1, 3, H, W]

        # ---------- 1. 语义分支 (Qwen2-VL ViT) ----------


        vit_pixel_values, image_grid_thw = self.vit_image_transform(image)
        print(f"DEBUG: vit_pixel_values max: {vit_pixel_values.abs().max()}")
        
        device = next(self.visiontower.parameters()).device
        dtype = next(self.visiontower.parameters()).dtype

        vit_pixel_values = vit_pixel_values.to(device=device, dtype=dtype)
        image_grid_thw = image_grid_thw.to(device=device)
        vit_grid_thw = image_grid_thw.to(device=device)

        # 1.3 一次 forward 拿特征
        vit_feats = self.visiontower(vit_pixel_values, grid_thw=image_grid_thw)  # [B, N_vit, D]



        # ---------- 2. 几何分支 (DINO) ----------
        dino_images = self.dino_transform(image)          # -> [B, C, H'', W'']
        print(f"DEBUG [DINO]: input images max: {dino_images.abs().max().item():.4f}")

        B, C, H, W = dino_images.shape
        patch_size = self.dinoTower.config.patch_size  # 例如 16


        patch_size = self.dinoTower.config.patch_size
        dino_h_tokens = dino_images.shape[2] // patch_size
        dino_w_tokens = dino_images.shape[3] // patch_size
        # 构造符合 Qwen2-VL 格式的 grid_thw: [T, H, W]
        # 注意：这里假设是单张图，如果是视频需要根据 B 调整，但在 VLA 中通常 B 放在外面
        dino_grid_thw = torch.tensor(
            [[1, dino_h_tokens, dino_w_tokens]], 
            device=dino_images.device, 
            dtype=torch.int32
        )

        num_tokens_per_image = (H // patch_size) * (W // patch_size)  # 每张图的 token 数
        cu_seqlens = torch.arange(0, B * num_tokens_per_image + 1, num_tokens_per_image, 
                                  device=dino_images.device,
                                  dtype=torch.int32 
                                  )
        max_seqlen = num_tokens_per_image


        dino_out = self.dinoTower(dino_images, cu_seqlens, max_seqlen)
        if torch.isnan(dino_out).any():
            print("❌ NaN detected inside dinoTower!")
            # 尝试强制修复 (仅用于调试)
            dino_out = torch.nan_to_num(dino_out, 0.0)
        # dino_feats = dino_out.last_hidden_state

        
        # dino_feats = self.dinoTower(pixel_values=dino_images)        # [B, N_dino, dino_dim]
        geometric_tokens = self.dinoProjector(dino_out)            # [B, N_dino, D]
        if torch.isnan(geometric_tokens).any():
            print("❌ NaN detected after dinoProjector!")

        # ---------- 3. 存储 grid 信息（用于构建 position_ids）----------
        # 注意：这里只存储单张图的 grid，如果是 batch，需要在调用处累积
        # 在 omni_vla.py 的 embed_prefix 中会累积这些信息
        
        # ---------- 4. 返回 ----------
        return {
            "semantic": vit_feats,      # [B, N_vit, D]
            "geometric": geometric_tokens,  # [B, N_dino, D]
            "vit_grid": vit_grid_thw[0],   # [1, 3] -> 用于 build_3d_position_ids
            "dino_grid": dino_grid_thw[0], # [1, 3] -> 用于 build_3d_position_ids
        }


    def embed_language_tokens(self, tokens: torch.Tensor):
        """Embed language tokens using G2VLM's language model."""
        return self.g2vlm.language_model.model.embed_tokens(tokens)
    
    def build_prefix(
        self,
        image: torch.Tensor,
        text_tokens: torch.Tensor,
    ):
        """
        return:
            prefix_embeds:     [B, N, D]
            prefix_token_type: [B, N]

            token_type == 0 → semantic expert
            token_type == 1 → geometric expert
            token_type == 2 → language expert

        """

        image_embeds = self.embed_image(image)

        semantic_tokens = image_embeds["semantic"]    # [B, Ns, D]
        geometric_tokens = image_embeds["geometric"]  # [B, Ng, D]
        text_embeds = self.embed_language_tokens(text_tokens)  # [B, T, D]

        # --- 🚀 关键诊断：看看是谁带毒 ---
        print(f"DEBUG: semantic_tokens max: {semantic_tokens.abs().max()}")
        print(f"DEBUG: geometric_tokens max: {geometric_tokens.abs().max()}")
        print(f"DEBUG: text_embeds max: {text_embeds.abs().max()}")

        prefix_embeds = torch.cat(
            [semantic_tokens, geometric_tokens, text_embeds],
            dim=1,
        )

        token_type_ids = self.build_prefix_token_type_ids(
            semantic_tokens,
            geometric_tokens,
            text_embeds,
        )

        return prefix_embeds, token_type_ids

    @staticmethod
    def _gated_residual(x, y, gate):
        """
        Applies gated residual connection with optional gate parameter. 
        
        Args:
            x: Input tensor (residual)
            y: Output tensor to be added
            gate: Optional gate tensor to modulate the addition
            
        Returns:
            x + y if gate is None, otherwise x + y * gate
        """
        if x is None and y is None:
            return None
        if x is None or y is None:
            return x if x is not None else y
        if gate is None:
            return x + y
        return x + y * gate
    
    
    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        """
        inputs_embeds:
            [0] prefix embeds  (semantic + geometric + text)
            [1] suffix embeds  (action tokens)
        """
        if adarms_cond is None:
            adarms_cond = [None, None]

        # --------------------------------------------------
        # Case 1: only prefix (encode / prefill)
        # --------------------------------------------------
        if inputs_embeds[1] is None:
            prefix_output = self.g2vlm.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0],
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None

        # --------------------------------------------------
        # Case 2: only suffix (decode action)
        # --------------------------------------------------
        elif inputs_embeds[0] is None:
            suffix_output = self.action_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1],
            )
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None

        # --------------------------------------------------
        # Case 3: prefix + suffix joint attention (PI-0 core)
        # --------------------------------------------------
        else:
            # 🔑 和原 PI-0 完全一致，只是换了 prefix model
            models = [
                self.g2vlm.language_model,   # prefix expert (semantic + geometric + text)
                self.action_expert.model,     # suffix expert (action)
            ]

            num_layers = models[0].config.num_hidden_layers

            # debug
            for i, x in enumerate(inputs_embeds):
                if x is not None:
                    print(f"Expert {i} input max: {x.abs().max()}")
            
            # 确保 grid 列表已初始化（在 embed_prefix 中会被填充）
            if not hasattr(self, 'current_vit_grid'):
                self.current_vit_grid = []
            if not hasattr(self, 'current_dino_grid'):
                self.current_dino_grid = []



            # 如果你没有真实的 full_input_ids，至少需要根据长度构造一个 LongTensor
            # 注意：必须是 Long 类型
            batch_size = inputs_embeds[0].shape[0]
            prefix_len = inputs_embeds[0].shape[1]
            suffix_len = inputs_embeds[1].shape[1]
            total_len = prefix_len + suffix_len

            # 如果 position_ids 为 None，使用存储的 grid 信息构建
            if position_ids is None:
                # 使用和 omni_vla.py 中相同的方法构建 3D position_ids
                from ..data_vlm.data_utils import get_rope_index_image_3D
                
                device = inputs_embeds[0].device
                b = batch_size
                curr_pos_val = 0
                
                # 1. 构建 ViT (语义) 位置编码
                all_vit_pos = []
                if hasattr(self, 'current_vit_grid') and len(self.current_vit_grid) > 0:
                    for grid in self.current_vit_grid:
                        pos_3d, delta = get_rope_index_image_3D(
                            grid.flatten()[:3] if grid.dim() > 0 else grid[:3], 
                            curr_position_id=curr_pos_val
                        )
                        all_vit_pos.append(pos_3d.unsqueeze(1).repeat(1, b, 1))
                        curr_pos_val += int(delta) + 1
                
                # 2. 构建 DINO (几何) 位置编码
                all_dino_pos = []
                if hasattr(self, 'current_dino_grid') and len(self.current_dino_grid) > 0:
                    for grid in self.current_dino_grid:
                        pos_3d, delta = get_rope_index_image_3D(
                            grid.flatten()[:3] if grid.dim() > 0 else grid[:3], 
                            curr_position_id=curr_pos_val
                        )
                        all_dino_pos.append(pos_3d.unsqueeze(1).repeat(1, b, 1))
                        curr_pos_val += int(delta) + 1
                
                # 3. 计算文本和动作的长度
                current_vision_len = sum([p.shape[-1] for p in all_vit_pos]) + sum([p.shape[-1] for p in all_dino_pos])
                actual_prefix_len = prefix_len
                text_len = actual_prefix_len - current_vision_len
                
                # 4. 构建文本和动作的位置编码（线性 T 轴）
                total_incremental_len = text_len + suffix_len
                incremental_ids = torch.arange(curr_pos_val, curr_pos_val + total_incremental_len, device=device)
                text_act_pos = incremental_ids.unsqueeze(0).unsqueeze(0).repeat(3, b, 1)
                
                # 5. 拼接所有位置编码
                all_pos = all_vit_pos + all_dino_pos
                if all_pos:
                    full_pos = torch.cat(all_pos + [text_act_pos], dim=-1)
                else:
                    # 如果没有视觉信息，只使用文本和动作
                    full_pos = text_act_pos
                
                position_ids = full_pos.to(device)
                
                # 验证长度
                expected_len = actual_prefix_len + suffix_len
                if position_ids.shape[-1] != expected_len:
                    logging.warning(
                        f"Position IDs length mismatch: got {position_ids.shape[-1]}, expected {expected_len}. "
                        f"Using fallback linear position encoding."
                    )
                    # 后备方案：使用简单的线性位置编码
                    position_ids = torch.arange(expected_len, device=device).unsqueeze(0).unsqueeze(0).repeat(3, b, 1)

            # 确保 position_ids 是 3 维的: [3, B, L]
            if position_ids.dim() == 2:
                position_ids = position_ids.unsqueeze(0).repeat(3, 1, 1)

            # gradient checkpointing（原样保留）
            use_gradient_checkpointing = (
                hasattr(self.action_expert.model, "gradient_checkpointing")
                and self.action_expert.model.gradient_checkpointing
                and self.training
            )

            if self.training and hasattr(self.action_expert.model, "gradient_checkpointing"):
                if not self.action_expert.model.gradient_checkpointing:
                    self.action_expert.model.gradient_checkpointing = True
                use_gradient_checkpointing = True

            def compute_layer_complete(
                layer_idx,
                inputs_embeds,
                attention_mask,
                position_ids,# 这里的 position_ids 必须是 [3, B, L] 的 LongTensor
                adarms_cond,
            ):
                query_states = []
                key_states = []
                value_states = []
                gates = []

                for i, hidden_states in enumerate(inputs_embeds):

                    layer = models[i].base_model.layers[layer_idx]
                    hidden_states = layer.input_layernorm(hidden_states)  # 不传 cond
                    # 创建全 1 gate，占位
                    gate = torch.full_like(hidden_states, 0.001)
                    gate = gate.to(hidden_states.dtype)
                        
                    device = layer.self_attn.q_proj.weight.device
                    dtype = layer.self_attn.q_proj.weight.dtype

                    hidden_states = hidden_states.to(device=device, dtype=dtype)
                    
                    gates.append(gate)

                    input_shape = hidden_states.shape[:-1]
                    hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

                    print(f"LayerNorm out max: {hidden_states.abs().max()}")

                    q = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    k = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    v = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                    print(f"Q proj max: {q.abs().max()}")

                    query_states.append(q)
                    key_states.append(k)
                    value_states.append(v)

                # 🔑 concat 前确认 hidden_size 对齐
                for i, x in enumerate(query_states):
                    assert x.shape[-1] == layer.self_attn.head_dim, f"Expert {i} Q shape mismatch"

                # --- 拼接所有专家的 Token ---
                # query_states 拼接后的形状: [B, num_heads, total_seq_len, head_dim]
                # concat attention
                query_states = torch.cat(query_states, dim=2)
                key_states = torch.cat(key_states, dim=2)
                value_states = torch.cat(value_states, dim=2)

                print("Fixed Query Shape:", query_states.shape)
                print("Fixed key_states Shape:", key_states.shape)
                print("Fixed value_states Shape:", value_states.shape)


                # 1. 获取 3D 旋转频率
                rope_module = models[0].base_model.layers[0].self_attn.rotary_emb

                prefix_len = inputs_embeds[0].shape[1]
                suffix_len = inputs_embeds[1].shape[1]
                total_len = prefix_len + suffix_len

                if position_ids.dim() == 2:
                    # 扩展为 [3, batch_size, seq_len]
                    # Qwen2-VL 期望第 0 维是 [T_index, H_index, W_index]
                    position_ids = position_ids.unsqueeze(0).repeat(3, 1, 1)

                # 2. 调用 rotary_emb 的 forward
                # 在 Qwen2-VL 中，rotary_emb(value_states, position_ids) 会：
                # a) 根据 position_ids (3, B, L) 提取 T, H, W 的索引
                # b) 针对 head_dim 的不同部分计算对应的旋转频率
                # c) 返回符合 M-RoPE 规则的 cos 和 sin
                with torch.no_grad():
                    # cos, sin 形状通常为 [3, B, L, head_dim // (某因子)] 
                    # 或者在最新版 HF 中直接返回拼接好的变换张量
                    cos, sin = rope_module(value_states, position_ids)
                    print(f"Cos max: {cos.max()}, Sin max: {sin.max()}") # 👈 检查这里


                # # 1. Handle M-RoPE 5D output (if it returns the 3-axis components)模型会丢失高度和宽度的空间坐标 
                # if cos.dim() == 5:
                #     # Most apply_rotary_pos_emb functions expect 4D.
                #     # Usually, we take index 0 or use a specific M-RoPE helper.
                #     cos = cos[0]
                #     sin = sin[0]

                # # 2. FORCE the head dimension to be 1 for broadcasting
                # # If shape is [Batch, 2, Seq, Dim], we want [Batch, 1, Seq, Dim]
                # if cos.shape[1] != 1:
                #     # We take only the first slice because RoPE is identical across heads
                #     cos = cos[:, :1, :, :]
                #     sin = sin[:, :1, :, :]

                print(f"Broadcast-ready Cos shape: {cos.shape}")

                # 2. 应用 3D M-RoPE
                query_states, key_states = apply_rotary_pos_emb_vision_3d(
                    query_states,
                    key_states,
                    cos,
                    sin
                )
                # q_embed, _ = apply_rotary_pos_emb(query_states, query_states, cos, sin)
                # _, k_embed = apply_rotary_pos_emb(key_states, key_states, cos, sin)
                # query_states = q_embed
                # key_states = k_embed

                print(f"Q max: {query_states.max()}, K max: {key_states.max()}")

                # query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
                #     query_states, key_states, cos, sin, unsqueeze_dim=1
                # )

                # 尝试获取，如果获取不到则手动计算
                attn_module = models[0].base_model.layers[layer_idx].self_attn
                if hasattr(attn_module, "scaling"):
                    scaling = attn_module.scaling
                else:
                    # head_dim 通常是 128 或 64
                    scaling = attn_module.head_dim ** -0.5

                print(query_states.shape)
                print(key_states.shape)
                print(value_states.shape)
                print(attention_mask.shape)

                # if query_states.dim() == 5:
                #     # Qwen2-VL 的 apply_mrope 可能会保留 3D 维度。
                #     # 实际上 M-RoPE 已经完成了旋转，我们只需要取其中一个分量或者对齐维度。
                #     # 在标准实现中，旋转是原位的，我们通过 view 把它压回 4 维。
                #     # 注意：这里取 query_states[0] 是不行的，因为三个分量分别旋转了不同的 head 部分。
                #     # 正确做法是 view 成 4 维，因为 num_heads 已经包含了所有的信息。
                    
                #     b_size = value_states.shape[0] # 真实的 Batch Size (1)
                    
                #     # 检查 query_states 的总维度是否匹配
                #     # 如果是 [3, B, H, L, D]，通常 Qwen 会在内部把 H 切分，
                #     # 但如果 apply_mrope 返回的是 5 维，说明它没有自动 squeeze。
                #     query_states = query_states[0]
                #     key_states = key_states[0]
                #     # query_states = query_states.view(batch_size, num_heads, seq_len, head_dim)
                #     # key_states = key_states.view(batch_size, num_heads, seq_len, head_dim)
                    

                #     # 再次打印确认，应该是 [1, 12, 1059, 128] 这种 4 维格式
                #     print("Fixed Query Shape:", query_states.shape)
                #     print("Fixed key Shape:", key_states.shape)

                # # 确保是 4D 且维度对齐
                # batch_size = value_states.shape[0]
                # seq_len = value_states.shape[2]

                # # 强制指定维度，防止 view 自动相乘
                # query_states = query_states.reshape(batch_size, 12, seq_len, 128)
                # key_states = key_states.reshape(batch_size, 2, seq_len, 128)

                # 打印一下确认：应该是 [1, 12, 1059, 128] 和 [1, 2, 1059, 128]
                print(f"Final Q: {query_states.shape}, K: {key_states.shape}")

                att_output, _ = modeling_gemma.eager_attention_forward(
                    models[0].base_model.layers[layer_idx].self_attn,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    scaling,
                )

                head_dim = models[0].base_model.layers[layer_idx].self_attn.head_dim
                num_heads = models[0].base_model.layers[layer_idx].self_attn.num_heads
                att_output = att_output.reshape(att_output.shape[0], -1, num_heads * head_dim)

                outputs = []
                start = 0
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].base_model.layers[layer_idx]
                    expert_dtype = layer.mlp.gate_proj.weight.dtype

                    end = start + hidden_states.shape[1]

                    out = layer.self_attn.o_proj(att_output[:, start:end])
                    out = self._gated_residual(hidden_states, out, gates[i])
                    

                    residual = out.clone()
                    # out, gate = layer.post_attention_layernorm(out, cond=adarms_cond[i])
                    out = layer.post_attention_layernorm(out)
                    gate = torch.ones_like(hidden_states)
                    out = out.to(expert_dtype)
                    out = layer.mlp(out)
                    out = self._gated_residual(residual, out, gate)

                    outputs.append(out)
                    start = end

                return outputs

            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        use_reentrant=False,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                    )

            # final norm
            outputs = []
            for i, hidden_states in enumerate(inputs_embeds):
                out = models[i].base_model.norm(hidden_states)
                outputs.append(out)

            prefix_output, suffix_output = outputs
            prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values
