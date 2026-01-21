
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
from transformers.models.qwen2_vl import modeling_qwen2_vl

from transformers.models.qwen2_vl import Qwen2VLForConditionalGeneration 
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel
from torch.nn.attention.flex_attention import create_block_mask

from openpi import models_pytorch as _mpp  # 官方
from openpi.models import gemma as _gemma  # 官方
from openpi.models_pytorch import preprocessing_pytorch as _pp  # 官方
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel  # 官方
from openpi.vlm_expert.dinov2_with_registers import Dinov2WithRegistersConfig
from openpi.vlm_expert.dinov2_with_registers import Dinov2WithRegistersModel

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


import logging
import sys
from pathlib import Path
from typing import Literal

import os

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
        
        "Reasoning module"
        reasoning_config_hf = CONFIG_MAPPING["qwen2_vl"]()
        reasoning_config_hf.text_config.hidden_size = reasoning_expert_config.hidden_size
        reasoning_config_hf.text_config.intermediate_size = reasoning_expert_config.intermediate_size
        reasoning_config_hf.text_config.num_attention_heads = reasoning_expert_config.num_attention_heads
        reasoning_config_hf.text_config.head_dim = reasoning_expert_config.head_dim
        reasoning_config_hf.text_config.num_hidden_layers = reasoning_expert_config.num_hidden_layers
        reasoning_config_hf.text_config.num_key_value_heads = reasoning_expert_config.num_key_value_heads
        reasoning_config_hf.text_config.max_position_embeddings = 262144
        reasoning_config_hf.text_config.rope_scaling = {
            "mrope_interleaved": True,
            "mrope_section": [24, 20, 20],
            "rope_type": "default"
        }
        reasoning_config_hf.text_config.tie_word_embeddings = True
        reasoning_config_hf.tie_word_embeddings = True
        reasoning_config_hf.vision_config.deepstack_visual_indexes=[5, 11, 17]
        reasoning_config_hf.vision_config.depth=24
        reasoning_config_hf.vision_config.hidden_size=1024
        reasoning_config_hf.vision_config.intermediate_size=4096
        reasoning_config_hf.vision_config.out_hidden_size=2048

        self.reasoning_module = Qwen2VLForConditionalGeneration(reasoning_config_hf)
