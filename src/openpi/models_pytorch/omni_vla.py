import logging
import math
from typing import Optional

import matplotlib.pyplot as plt

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
import torch._dynamo

import openpi.models.gemma as _gemma
from openpi.models_pytorch.g2vlm_pi0_pytorch import G2VLMWithActorExpertModel
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing
from openpi.models_pytorch.vlm_with_spatial import VLMWithSpatialActionExpertModel
from openpi.vggt.models.vggt import VGGT
from openpi.vlm_expert.g2vlm.g2vlm import G2VLMConfig
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.omni_config import OmniConfig
from transformers.cache_utils import DynamicCache
from einops import rearrange
import copy

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

def _debug_tensor(name, x, max_print=5):
    if x is None:
        print(f"{name}: None")
        return
    print(
        f"{name}: "
        f"shape={tuple(x.shape)}, "
        f"dtype={x.dtype}, "
        f"device={x.device}"
    )
    # Only print the beginning to avoid cluttering
    print(f"{name} sample:", x.flatten()[:max_print].tolist())
    print("-" * 60)


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

        self.action_horizon = config.action_horizon

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


        # export TORCHDYNAMO_DISABLE=1


        self.action_in_proj = nn.Linear(32, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, 32)


        self.state_proj = nn.Linear(32, action_expert_config.width)
        self.action_time_mlp_in = nn.Linear(
            2 * action_expert_config.width, action_expert_config.width
        )
        self.action_time_mlp_out = nn.Linear(
            action_expert_config.width, action_expert_config.width
        )

        # In the inference phase, keep the logic correct and no longer wrap bound methods with torch.compile,
        # otherwise it will lead to lost binding relationships.
        torch.set_float32_matmul_precision("high")

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        # Hardcoded freeze
        for param in self.reasoning_spatial_expert.vggt_encoder.parameters():
            param.requires_grad = False

        # Freeze VLM
        # for param in self.reasoning_spatial_expert.reasoning_expert.parameters():
        #     param.requires_grad = False

        for param in self.reasoning_spatial_expert.reasoning_expert.vision_tower.parameters():
            param.requires_grad = False

        # Freeze spatial
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
        """Set requires_grad for each expert according to config."""
        if getattr(self.config, "freeze_vision_encoder", False):
            self.reasoning_spatial_expert.reasoning_expert.vision_tower.eval()
            for p in self.reasoning_spatial_expert.reasoning_expert.vision_tower.parameters():
                p.requires_grad = False

        if getattr(self.config, "train_expert_only", False):
            self.reasoning_spatial_expert.reasoning_expert.eval()
            for p in self.reasoning_spatial_expert.reasoning_expert.parameters():
                p.requires_grad = False
            self.reasoning_spatial_expert.spatial_expert.eval()
            for p in self.reasoning_spatial_expert.spatial_expert.parameters():
                p.requires_grad = False

        if getattr(self.config, "train_vlm_only", False):
            self.reasoning_spatial_expert.reasoning_expert.eval()
            for p in self.reasoning_spatial_expert.reasoning_expert.parameters():
                p.requires_grad = False
            self.reasoning_spatial_expert.spatial_expert.eval()
            for p in self.reasoning_spatial_expert.spatial_expert.parameters():
                p.requires_grad = False
            self.reasoning_spatial_expert.action_expert.eval()
            for p in self.reasoning_spatial_expert.action_expert.parameters():
                p.requires_grad = False
        
    
    def train(self, mode: bool = True):
        super().train(mode)

        # if self.config.freeze_vision_encoder:
        #     self.reasoning_spatial_expert.reasoning_expert.vision_tower.eval()

        # if self.config.train_expert_only:
        #     self.reasoning_spatial_expert.reasoning_expert.eval()
        #     self.reasoning_spatial_expert.spatial_expert.eval()

        # if self.config.freeze_VGGT_model:
        #     self.reasoning_spatial_expert.vggt_encoder.eval()
        
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

    
    def embed_spatial(self, images, img_masks):
        """
        Embed spatial images, ensuring the Mask length is strictly aligned with the number of VGGT output tokens.
        """
        # 1. Prepare image tensor [B, S, C, H, W]
        images_tensor = torch.stack(images, dim=1)
        B, S, C, H, W = images_tensor.shape

        # 2. VGGT specific preprocessing: resize to 256x256, normalize to [-1, 1]
        # Reference prepare_spatial_features
        vggt_dtype = next(self.reasoning_spatial_expert.vggt_encoder.parameters()).dtype
        images_flat = images_tensor.reshape(B * S, C, H, W)
        images_flat = F.interpolate(images_flat, size=(252, 252), mode="bilinear", align_corners=False)
        images_flat = images_flat * 2 - 1  # [0,1] -> [-1,1]
        images_tensor = images_flat.reshape(B, S, C, 252, 252).to(dtype=vggt_dtype)

        # 3. VGGT feature extraction
        def image_embed_func(img):
            # Input [B, S, C, H, W], returns dict where features[-1] is the last layer features
            # Shape is usually [B, S, N, D], where N is the number of tokens
            res = self.reasoning_spatial_expert.vggt_encoder(img)
            return res["features"][-1]

        with torch.no_grad():
            img_emb = image_embed_func(images_tensor) # [B, S, N, D]
        
        # Get the actual number of Tokens N
        B, S, N, D = img_emb.shape
        device = img_emb.device

        # 3. Dynamically build Mask
        # Logic: In the N tokens output by VGGT, the first part is usually Patch, and the latter is Global
        # To be rigorous, we directly generate an all-1 Mask based on the actual shape of img_emb [B, S*N]
        # If fine-grained padding is needed for image regions later, it is recommended to slice here based on N
        
        # Generate pad_masks [B, S*N] that are completely consistent with the length of img_emb
        # We first generate [B, S, N] and then view it as [B, S*N] to keep the dimensional semantics clear
        pad_masks = torch.ones((B, S, N), dtype=torch.bool, device=device)
        
        # If original img_masks (for batch padding of images) need to take effect:
        if img_masks is not None:
            if isinstance(img_masks, list):
                mask_tensor = torch.stack(img_masks, dim=1).to(device)  # [B, S]
            else:
                mask_tensor = img_masks.to(device)
                
            # Broadcast [B, S] image-level mask to [B, S, N]
            # This means if an image is padded, all N tokens corresponding to it will be masked
            pad_masks = pad_masks * mask_tensor.unsqueeze(-1)

        # 4. Flatten sequence to [B, S*N, D]
        img_emb = img_emb.reshape(B, S * N, D)
        pad_masks = pad_masks.reshape(B, S * N)
        
        # 5. Build attention mask
        # Set the first token to 1 to establish a causal boundary, so that prefix cannot attend to middle,
        # but middle can attend to prefix. This way during inference, prefix is processed first
        # consistent with training behavior, eliminating train-inference gap.
        # Reference embed_middle: att_masks = [1] + [0] * (seq_len - 1)
        seq_len = pad_masks.shape[1]
        att_masks = torch.zeros((B, seq_len), dtype=torch.bool, device=device)
        att_masks[:, 0] = True  # Set the first token to 1 to establish a causal boundary

        # 6. Dimension mapping to inference model dimension
        img_emb = self.spatial_to_reasoning(img_emb)

        return img_emb, pad_masks, att_masks


    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        target_dtype = self.state_proj.weight.dtype
        state = state.to(target_dtype)
        noisy_actions = noisy_actions.to(target_dtype)

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
        att_masks += [1] + ([0] * (self.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        # Use bool consistent with prefix/middle, otherwise torch.cat will error due to dtype mismatch
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def get_position_ids(self, pad_masks):
        # pad_masks: [batch_size, total_seq_len] 
        # Here total_seq_len is the sum of prefix + middle + suffix
        
        device = pad_masks.device
        seq_len = pad_masks.shape[1]
        
        # Official PaliGemma logic: whether it's image or text, line up in order
        # Generate 0, 1, 2, ..., total_seq_len - 1
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand_as(pad_masks)
        
        # Note: if pad_masks contains padding, no need to specifically fill 0 here
        # Because RoPE will generate rotation for each position, and the real 'masking' is achieved by Attention Mask
        return position_ids

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
        """Use VGGT for spatial features (consistent with embed_spatial); can be extended here if other backbones are needed."""
        images = torch.stack([batch[f"{OBS_IMAGES}.image{i}"] for i in range(3)], dim=1)  # B, N_view, T, C, H, W
        B, N_view, T = images.shape[:3]
        images = rearrange(images, "b n t c h w -> (b n t) c h w")
        images = F.interpolate(images, size=(256, 256), mode="bilinear", align_corners=False)
        images = images * 2 - 1
        # VGGT requires [B, S, C, H, W]
        images_5d = rearrange(images, "(b n t) c h w -> b (n t) c h w", b=B, n=N_view, t=T)
        with torch.no_grad():
            res = self.reasoning_spatial_expert.vggt_encoder(images_5d)
        features = res["features"][-1]  # [B, N_view*T, N, D]
        features = rearrange(features, "b (n t) N d -> b n t N d", n=N_view, t=T)
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
        position_ids = self.get_position_ids(pad_masks)

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

        # Step length temporarily set, can be changed to dynamic configuration later
        suffix_out = suffix_out[:, -self.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        u_t = u_t[:, -self.action_horizon :]


        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        loss_action = F.mse_loss(u_t, v_t, reduction="none")

        return loss_action
    

    @torch._dynamo.disable
    @torch.no_grad()
    def sample_actions(self, observation, device=None, noise=None, num_steps: int = 10) -> Tensor:
        """
        Inference sampling: Use KV cache to accelerate inference.
        1. Process prefix (language+image) first to get past_key_values
        2. Process middle (spatial features) next, reusing prefix's KV cache
        3. In the denoising loop, only process suffix each time, reusing prefix and middle's KV cache
        """
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(
            observation, train=False
        )

        bsize = state.shape[0]
        device = state.device

        if noise is None:
            # Internally always denoising in the 32-dimensional action space, length is action_horizon
            actions_shape = (bsize, self.action_horizon, 32)
            noise = self.sample_noise(actions_shape, device)

        target_dtype = self.action_in_proj.weight.dtype
        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=target_dtype, device=device)
        x_t = noise.to(target_dtype)
        time = torch.tensor(1.0, dtype=target_dtype, device=device)

        # All three experts use eager implementation to avoid differences in implementations like flash-attn
        self.reasoning_spatial_expert.reasoning_expert.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        self.reasoning_spatial_expert.spatial_expert.config._attn_implementation = "eager"  # noqa: SLF001
        self.reasoning_spatial_expert.action_expert.config._attn_implementation = "eager"  # noqa: SLF001

        # 1. Process prefix (language+image)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = self.get_position_ids(prefix_pad_masks)
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)

        # Undo the normalizer scaling inside GemmaModel during inference to align with training logic
        normalizer = torch.tensor(prefix_embs.shape[-1]**0.5, dtype=prefix_embs.dtype, device=prefix_embs.device)
        prefix_embs_unscaled = prefix_embs / normalizer

        # Process prefix, get past_key_values
        _, past_key_values = self.reasoning_spatial_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs_unscaled, None, None],
            use_cache=True,
        )
        max_prefix_position_ids = prefix_position_ids.max(dim=-1, keepdim=True).values

        # 2. Process middle (spatial features)
        middle_embs, middle_pad_masks, middle_att_masks = self.embed_spatial(
            images, img_masks
        )

        middle_len = middle_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        
        # Build attention mask for prefix + middle
        # Ensure middle's own pad_masks and prefix's pad_masks work together
        prefix_pad_2d_masks = middle_pad_masks[:, :, None] & prefix_pad_masks[:, None, :]
        middle_att_2d_masks = make_att_2d_masks(middle_pad_masks, middle_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, middle_att_2d_masks], dim=2)

        # middle's position_ids start from max_prefix_position_ids + 1
        # For calling spatial_expert individually, a shape of [batch_size, middle_len] is needed
        # Index 1 corresponds to spatial_expert (reasoning=0, spatial=1, action=2)
        middle_position_ids_2d = torch.arange(1, middle_len + 1, dtype=torch.long, device=device)
        middle_position_ids_2d = middle_position_ids_2d.unsqueeze(0).expand(batch_size, -1)  # [batch_size, middle_len]
        middle_position_ids_2d = middle_position_ids_2d + max_prefix_position_ids  # [batch_size, 1] broadcasts to [batch_size, middle_len]

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)

        # Undo normalizer scaling inside GemmaModel during inference
        normalizer_m = torch.tensor(middle_embs.shape[-1]**0.5, dtype=middle_embs.dtype, device=middle_embs.device)
        middle_embs_unscaled = middle_embs / normalizer_m

        # Process middle, reuse prefix's KV cache
        # Note: When calling expert individually, position_ids should be of shape [batch_size, seq_len]
        (_, middle_out, _), past_key_values = self.reasoning_spatial_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=middle_position_ids_2d,
            past_key_values=past_key_values,
            inputs_embeds=[None, middle_embs_unscaled, None],
            use_cache=True,
        )

        # Save max_position_ids for subsequent position_ids calculation of suffix
        max_position_ids = middle_position_ids_2d.max(dim=-1, keepdim=True).values  # [batch_size, 1]
        curr_pad_masks = torch.cat([prefix_pad_masks, middle_pad_masks], dim=1)

        # 3. Denoising loop: only process suffix each time, reusing prefix and middle's KV cache
        while time >= -dt / 2:
            expanded_time = time.expand(bsize).to(target_dtype)
            v_t = self.denoise_step(
                state=state,
                prefix_pad_masks=curr_pad_masks,
                past_key_values=past_key_values,
                max_position_ids=max_position_ids,
                x_t=x_t.to(target_dtype),
                timestep=expanded_time.to(torch.float32),
            )
            x_t = x_t + dt * v_t
            time = (time + dt).to(target_dtype)

        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        max_position_ids,
        x_t,
        timestep,
    ):
        """Single step denoising: only process suffix, reuse prefix and middle's KV cache."""
        # 1) Only generate suffix embedding
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            state, x_t, timestep
        )

        # Dtype processing consistent with the training phase
        if (
            self.reasoning_spatial_expert.reasoning_expert.language_model.model.layers[
                0
            ].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

        # 2) Build attention mask
        # When using KV cache, transformers will automatically handle attention mask
        # We only need to build attention mask for the new sequence (suffix)
        # Transformers internally will automatically extend it to the total length including cached sequence
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        
        # Only build attention mask for suffix
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        
        # When using KV cache, a complete attention mask including cached sequence needs to be built
        # Get actual cached sequence length from past_key_values
        if past_key_values is not None:
            cached_seq_len = past_key_values.get_seq_length()
            # To prevent modifying the reused cache in the denoising loop
            from transformers.cache_utils import DynamicCache
            past_key_values_copy = DynamicCache()
            past_key_values_copy.key_cache = list(past_key_values.key_cache)
            past_key_values_copy.value_cache = list(past_key_values.value_cache)
            if hasattr(past_key_values, '_seen_tokens'):
                past_key_values_copy._seen_tokens = past_key_values._seen_tokens
        else:
            cached_seq_len = 0
            past_key_values_copy = None
        
        # Build complete attention mask: [batch_size, suffix_len, cached_seq_len + suffix_len]
        if cached_seq_len > 0:
            # Consider padding in the cached sequence
            cached_mask = suffix_pad_masks[:, :, None] & prefix_pad_masks[:, None, :cached_seq_len]
            full_att_2d_masks = torch.cat([cached_mask, suffix_att_2d_masks], dim=2)
        else:
            full_att_2d_masks = suffix_att_2d_masks

        # suffix's position_ids start from max_position_ids + 1
        # For calling action_expert individually, a shape of [batch_size, suffix_len] is needed
        # Index 2 corresponds to action_expert (reasoning=0, spatial=1, action=2)
        device = suffix_embs.device
        position_ids = torch.arange(1, suffix_len + 1, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)  # [batch_size, suffix_len]
        position_ids = position_ids + max_position_ids  # [batch_size, 1] broadcasts to [batch_size, suffix_len]

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)

        # Undo normalizer scaling inside GemmaModel during inference
        normalizer_s = torch.tensor(suffix_embs.shape[-1]**0.5, dtype=suffix_embs.dtype, device=suffix_embs.device)
        suffix_embs_unscaled = suffix_embs / normalizer_s

        # 3) Only process suffix, reuse prefix and middle's KV cache
        outputs_embeds, _ = self.reasoning_spatial_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values_copy,
            inputs_embeds=[None, None, suffix_embs_unscaled],
            use_cache=False,  # suffix changes every time, no cache needed
        )

        # 4) Extract the last action_horizon tokens consistent with training loss, and project back to action space
        suffix_out = outputs_embeds[2]
        suffix_out = suffix_out[:, -self.action_horizon :]
        suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)
        v_t = self.action_out_proj(suffix_out)
        return v_t