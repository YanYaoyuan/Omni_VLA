import logging
import math
from typing import Optional

import matplotlib.pyplot as plt

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

import openpi.models.gemma as _gemma
from openpi.models_pytorch.g2vlm_pi0_pytorch import G2VLMWithActorExpertModel
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing
from openpi.models_pytorch.vlm_with_spatial import VLMWithSpatialActionExpertModel
from openpi.vggt.models.vggt import VGGT
from openpi.vlm_expert.g2vlm.g2vlm import G2VLMConfig
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.omni_config import OmniConfig

from einops import rearrange

from ..data_vlm.data_utils import get_rope_index_image_3D
from ..data_vlm.data_utils import get_rope_index_image_3D_dino

OBS_STR = "observation"
OBS_PREFIX = OBS_STR + "."
OBS_ENV_STATE = OBS_STR + ".environment_state"
OBS_STATE = OBS_STR + ".state"
OBS_IMAGE = OBS_STR + ".image"
OBS_IMAGES = OBS_IMAGE + "s"
OBS_LANGUAGE = OBS_STR + ".language"
OBS_LANGUAGE_TOKENS = OBS_LANGUAGE + ".tokens"
OBS_LANGUAGE_ATTENTION_MASK = OBS_LANGUAGE + ".attention_mask"

ACTION = "action"

def pad_vector(vector, new_dim):
    """Pad the last dimension of a vector to new_dim with zeros.

    Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))

def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: Tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    device = torch.device(device)
    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks





class OmniVLA(nn.Module):
    def __init__(self, config: OmniConfig, device: torch.device):
        """
        Initialize Omni VLA
        """
        super().__init__()
        self.config = config
        spatial_config = _gemma.get_config(config.spatial_expert_variant)
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.reasoning_spatial_expert = VLMWithSpatialActionExpertModel(
            paligemma_config,
            spatial_config,
            action_expert_config,
            precision=config.dtype,
        )

        dtype = next(self.reasoning_spatial_expert.vggt_encoder.parameters()).dtype
        self.spatial_to_reasoning = torch.nn.Linear(2048, 1024, dtype=dtype)





        self.action_in_proj = nn.Linear(32, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, 32)


        self.state_proj = nn.Linear(32, action_expert_config.width)
        self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
        self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")
        self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        # 硬编码冻结
        for param in self.reasoning_spatial_expert.vggt_encoder.parameters():
            param.requires_grad = False

        # 冻结VLM
        for param in self.reasoning_spatial_expert.reasoning_expert.parameters():
            param.requires_grad = False

        # 冻结spatial
        # for param in self.reasoning_spatial_expert.spatial_expert.parameters():
        #     param.requires_grad = False

        # msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        # try:
        #     from transformers.models.siglip import check

        #     if not check.check_whether_transformers_replace_is_installed_correctly():
        #         raise ValueError(msg)
        # except ImportError:
        #     raise ValueError(msg) from None
        

    def set_requires_grad(self):
        if self.config.freeze_vision_encoder:
            self.reasoning_spatial_expert.visual.eval()
            for params in self.reasoning_spatial_expert.und_expert.visual.parameters():
                params.requires_grad = False

        if self.config.train_expert_only:
            self.reasoning_spatial_expert.und_expert.eval()
            for params in self.reasoning_spatial_expert.und_expert.parameters():
                params.requires_grad = False
        
        if self.config.train_vlm_only:
            self.reasoning_spatial_expert.gen_expert.eval()
            for params in self.reasoning_spatial_expert.gen_expert.parameters():
                params.requires_grad = False
            self.reasoning_spatial_expert.act_expert.eval()
            for params in self.reasoning_spatial_expert.act_expert.parameters():
                params.requires_grad = False
        
    
    def train(self, mode: bool = True):
        super().train(mode)

        if self.config.freeze_vision_encoder:
            self.reasoning_spatial_expert.reasoning_expert.vision_tower.eval()

        if self.config.train_expert_only:
            self.reasoning_spatial_expert.reasoning_expert.eval()
            self.reasoning_spatial_expert.spatial_expert.eval()

        if self.config.freeze_VGGT_model:
            self.reasoning_spatial_expert.vggt_encoder.eval()
        
        return self
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.reasoning_spatial_expert.reasoning_expert.language_model.gradient_checkpointing = True
        self.reasoning_spatial_expert.reasoning_expert.vision_tower.gradient_checkpointing = True
        self.reasoning_spatial_expert.spatial_expert.model.gradient_checkpointing = True
        self.reasoning_spatial_expert.action_expert.model.gradient_checkpointing = True
        logging.info("Enabled gradient checkpointing for QwenA1 model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.reasoning_spatial_expert.reasoning_expert.language_model.gradient_checkpointing = False
        self.reasoning_spatial_expert.reasoning_expert.vision_tower.gradient_checkpointing = False
        self.reasoning_spatial_expert.spatial_expert.model.gradient_checkpointing = False
        self.reasoning_spatial_expert.action_expert.model.gradient_checkpointing = False
        logging.info("Disabled gradient checkpointing for QwenA1 model")

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.reasoning_spatial_expert.reasoning_expert.get_image_features(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)

            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.reasoning_spatial_expert.reasoning_expert.language_model.get_input_embeddings()(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def get_cosmos_features(self, images):
        shape = images.shape[:-3]
        c, h, w = images.shape[-3:]
        images = images.reshape(-1, c, h, w)
        images = F.interpolate(images, size=(256, 256), mode="bilinear", align_corners=False)
        images = images * 2 - 1  # [-1, 1]
        features = self.cosmos.encode(images)
        c, h, w = features.shape[-3:]
        features = features.view(*shape, c, h, w)
        return features

    def embed_spatial_old(self, images, img_masks):
        """Embed spatial images to prepare for Expert Gemma processing."""
        # 1. 准备图像张量 [B, S, C, H, W]
        images_tensor = torch.stack(images, dim=1)
        images_tensor = images_tensor.to(dtype=next(self.reasoning_spatial_expert.vggt_encoder.parameters()).dtype)
        B, S, C, H, W = images_tensor.shape

        # 2. 修复 Mask 处理逻辑
        # img_masks 现在是一个 list, 里面每个元素是 [B, S] 的 bool
        # 我们需要的是 [B, S, H, W]
        if isinstance(img_masks, list):
            # 即使堆叠后也只是 [B, S]，没有 H, W 维度
            mask_tensor = torch.stack(img_masks, dim=1).to(images_tensor.device) # [B, S]
        else:
            mask_tensor = img_masks.to(images_tensor.device)

        # 重点：将 [B, S] 扩展到 [B, S, H, W]
        # 使用 .unsqueeze(-1).unsqueeze(-1) 增加维度，然后用 .expand 填充空间
        full_spatial_masks = mask_tensor.unsqueeze(-1).unsqueeze(-1).expand(B, S, H, W)

        # 3. 接下来进行正常的下采样逻辑 (Patch Size = 14)
        gh, gw = H // 14, W // 14
        
        # 将 [B, S, H, W] -> [B*S, 1, H, W]
        flat_masks = full_spatial_masks.reshape(B * S, 1, H, W).float()
        
        # 下采样到 Patch 级别
        down_masks = torch.nn.functional.interpolate(flat_masks, size=(gh, gw), mode='nearest')
        
        # 最终对齐到 Token 数量: [B, S * gh * gw]
        pad_masks = down_masks.view(B, S * gh * gw).to(torch.bool)

        # 4. VGGT 特征提取
        def image_embed_func(img):
            # 注意：这里传入的 img 是 [B, S, C, H, W]
            res = self.reasoning_spatial_expert.vggt_encoder(img)
            # 确保返回的是视觉 token 序列
            return res["features"][-1] 

        img_emb = self._apply_checkpoint(image_embed_func, images_tensor)
        B_t, S_t, N_t, D_t = img_emb.shape
        img_emb = img_emb.view(B_t, S_t * N_t, D_t) 
        # 这里的 att_masks 通常和 pad_masks 一致

        att_masks = torch.zeros_like(pad_masks, dtype=torch.bool)

        return img_emb, pad_masks, att_masks
    
    def embed_spatial(self, images, img_masks):
        """Embed spatial images，输出与 embed_prefix 完全兼容"""
        # 1. 准备图像张量 [B, S, C, H, W]
        images_tensor = torch.stack(images, dim=1)
        images_tensor = images_tensor.to(
            dtype=next(self.reasoning_spatial_expert.vggt_encoder.parameters()).dtype
        )
        B, S, C, H, W = images_tensor.shape

        # 2. 修复 Mask 处理逻辑
        if isinstance(img_masks, list):
            mask_tensor = torch.stack(img_masks, dim=1).to(images_tensor.device)  # [B, S]
        else:
            mask_tensor = img_masks.to(images_tensor.device)

        # 扩展到 [B, S, H, W]
        full_spatial_masks = mask_tensor.unsqueeze(-1).unsqueeze(-1).expand(B, S, H, W)

        # 3. 下采样到 Patch 级别 (Patch Size = 14)
        gh, gw = H // 14, W // 14
        flat_masks = full_spatial_masks.reshape(B * S, 1, H, W).float()
        down_masks = torch.nn.functional.interpolate(flat_masks, size=(gh, gw), mode='nearest')
        pad_masks = down_masks.view(B, S * gh * gw).to(torch.bool)

        # 4. VGGT 特征提取
        def image_embed_func(img):
            # 输入 [B, S, C, H, W]，返回 [B, S, num_tokens, emb_dim]
            res = self.reasoning_spatial_expert.vggt_encoder(img)
            return res["features"][-1]  # [B, S, num_tokens, emb_dim]

        img_emb = self._apply_checkpoint(image_embed_func, images_tensor)  # [B, S, N, D]

        # 5. Flatten 序列到 [B, seq_len, emb_dim]
        B, S, N, D = img_emb.shape
        img_emb = img_emb.view(B, S * N, D)  # [B, S*N, D]

        # 6. 构建 attention mask，全 attention
        att_masks = torch.zeros_like(pad_masks, dtype=torch.bool)  # [B, S*N]

        img_emb = self.spatial_to_reasoning(img_emb)  # [B, seq_len, 1024]


        return img_emb, pad_masks, att_masks


    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if self.state_proj.weight.dtype == torch.float32:
            state = state.to(torch.float32)

        def state_proj_func(state):
            return self.state_proj(state)

        state_emb = self._apply_checkpoint(state_proj_func, state)
        embs.append(state_emb[:, None, :])
        bsize = state_emb.shape[0]
        device = state_emb.device

        state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)
        att_masks += [1]

        # Embed timestep using sine-cosine positional encoding
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=4e-3,
            max_period=4.0,
            device=timestep.device,
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        def mlp_func(action_time_emb):
            x = self.action_time_mlp_in(action_time_emb)
            x = F.silu(x)
            return self.action_time_mlp_out(x)

        action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)

        embs.append(action_time_emb)
        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def get_position_ids(self, lang_tokens, image_grid_thw, pad_masks): 
        L = lang_tokens.shape[1]
        pseudo_avail_token_id = 777
        padded_lang_tokens = torch.ones_like(pad_masks).to(lang_tokens) * pseudo_avail_token_id
        padded_lang_tokens[:, :L] = lang_tokens
        attention_mask = pad_masks.to(lang_tokens)
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.view(-1, 3)
        position_ids, rope_deltas = self.reasoning_spatial_expert.und_expert.model.get_rope_index(
            padded_lang_tokens, 
            image_grid_thw, 
            attention_mask=attention_mask, 
        )
        return position_ids, rope_deltas

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )
    
    def _preprocess_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Preprocess images for the model.
        """
        images = []
        img_masks = []

        for img_idx in range(3):
            img = batch[f"{OBS_IMAGES}.image{img_idx}"]
            mask = batch[f"{OBS_IMAGES}.image{img_idx}_mask"]

            images.append(img)
            img_masks.append(mask)
        
        images = torch.stack(images, dim=1)  # B, N_view, T, C, H, W
        img_masks = torch.stack(img_masks, dim=1)

        return images, img_masks
    
    def prepare_state(self, batch):
        """Pad state"""
        state = pad_vector(batch[OBS_STATE], self.config.max_state_dim)
        return state

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions
    
    def prepare_spatial_features(self, batch):
        images = torch.stack([batch[f"{OBS_IMAGES}.image{i}"] for i in range(3)], dim=1)  # B, N_view, T, C, H, W
        B, N_view, T = images.shape[:3]
        images = rearrange(images, 'b n t c h w -> (b n t) c h w')
        images = F.interpolate(images, size=(256, 256), mode="bilinear", align_corners=False)
        images = images * 2 - 1  # [-1, 1]
        features = self.model.cosmos.encode(images)
        features = rearrange(features, '(b n t) c h w -> b n t c h w', b=B, n=N_view, t=T)
        return features

    def forward(
        self, observation, actions, noise=None, time=None
    ) -> Tensor:
        """Do a full training forward pass and compute the loss."""
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)
        
        
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        middle_embs, middle_pad_masks, middle_att_masks = self.embed_spatial(
            images, img_masks,  
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(state, x_t, time)

        if (
            self.reasoning_spatial_expert.reasoning_expert.language_model.model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            middle_embs = middle_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, middle_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, middle_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        # position_ids, rope_deltas = self.get_position_ids(lang_tokens, image_grid_thw, pad_masks)
        # 先用PI0
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def forward_func(prefix_embs, middle_embs, suffix_embs, att_2d_masks_4d, position_ids):
            (_, middle_out, suffix_out), _ = self.reasoning_spatial_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, middle_embs, suffix_embs],
                use_cache=False,
            )
            return middle_out, suffix_out

        middle_out, suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, middle_embs, suffix_embs, att_2d_masks_4d, position_ids
        )

        # def cosmos_out_func(middle_out):
        #     return self.decode_cosmos(middle_out)
        
        # pred_cosmos_features = self._apply_checkpoint(cosmos_out_func, middle_out.to(dtype=torch.float32))

        # future_embs = self.get_cosmos_features(images[:, :, 2])
        # loss_gen = F.mse_loss(pred_cosmos_features[img_masks], future_embs.to(dtype=torch.float32)[img_masks])

        # 步长36临时设定，后续可改为动态配置
        suffix_out = suffix_out[:, -36 :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        u_t = u_t[:, -36 :]


        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        loss_action = F.mse_loss(u_t, v_t, reduction="none")

        return loss_action

    @torch.no_grad()  # see openpi `sample_actions` (slightly adapted)
    def sample_actions(
        self, images, img_masks, pixel_values, image_grid_thw, lang_tokens, lang_masks, state, noise=None, num_steps=None, decode_image=False
    ) -> Tensor:
        """Do a full inference forward and compute the action."""
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        bsize = state.shape[0]
        device = state.device
        dtype = state.dtype

        if noise is None:
            # Sample noise with padded dimension as expected by action_in_proj
            actions_shape = (
                bsize,
                self.config.chunk_size,
                self.config.max_action_dim,
            )  # Use config max_action_dim for internal processing
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            pixel_values, image_grid_thw, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids, rope_deltas = self.get_position_ids(lang_tokens, image_grid_thw, prefix_pad_masks)

        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.reasoning_spatial_expert.und_expert.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.reasoning_spatial_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None, None],
            use_cache=True,
        )
        max_prefix_position_ids = prefix_position_ids.max(dim=-1, keepdim=True).values

        middle_embs, middle_pad_masks, middle_att_masks = self.embed_middle(
            images[:, :, :2], img_masks, 
        )

        middle_len = middle_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, middle_len, prefix_len)
        middle_att_2d_masks = make_att_2d_masks(middle_pad_masks, middle_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, middle_att_2d_masks], dim=2)

        middle_position_ids = torch.arange(1, middle_len + 1).repeat(3, 1, 1).to(max_prefix_position_ids) + max_prefix_position_ids

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.reasoning_spatial_expert.gen_expert.config._attn_implementation = "eager"  # noqa: SLF001

        (_, middle_out, _), past_key_values = self.reasoning_spatial_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=middle_position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, middle_embs, None],
            use_cache=True,
        )

        max_position_ids = middle_position_ids.max(dim=-1, keepdim=True).values
        curr_pad_masks = torch.cat([prefix_pad_masks, middle_pad_masks], dim=1)

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                curr_pad_masks,
                past_key_values,
                max_position_ids, 
                x_t.to(dtype),
                expanded_time.to(dtype),
            )
            x_t = x_t + dt * v_t
            time += dt

        if decode_image:
            def cosmos_out_func(middle_out):
                return self.decode_cosmos(middle_out)
            pred_cosmos_features = self._apply_checkpoint(cosmos_out_func, middle_out.to(dtype=torch.bfloat16))
            pred_cosmos_features = pred_cosmos_features.squeeze(0)
            recon_images = self.cosmos.decode(pred_cosmos_features.squeeze(0))
        else:
            recon_images = None

        return x_t, recon_images

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        max_prefix_position_ids, 
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        position_ids = torch.arange(1, suffix_len + 1).repeat(3, 1, 1).to(max_prefix_position_ids) + max_prefix_position_ids

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.reasoning_spatial_expert.act_expert.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.reasoning_spatial_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, None, suffix_embs],
            use_cache=False,
        )

        suffix_out = outputs_embeds[2]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)