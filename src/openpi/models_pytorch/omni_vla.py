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
from openpi.vlm_expert.g2vlm.g2vlm import G2VLMConfig
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.omni_config import OmniConfig

from ..data_vlm.data_utils import get_rope_index_image_3D
from ..data_vlm.data_utils import get_rope_index_image_3D_dino

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
        Initialize from PI0Pytorch model config.
        
        Args:
            config: Configuration object
            g2vlm_model: Optional pre-initialized G2VLM model. If provided, will use G2VLM instead of PaliGemma.
        """
        super().__init__()
        self.config = config
        self.use_pre_g2vlm = config.use_pretrained_g2vlm

        action_expert_config = _gemma.get_config(config.action_expert_variant)
        g2_path = config.pretrained_g2vlm_path
        g2_config_path = config.g2vlm_config_path

        

        # Try to import G2VLM adapter
        try:
            from ..models_pytorch.g2vlm_pi0_pytorch import G2VLMWithActorExpertModel
            G2VLM_AVAILABLE = True
        except ImportError:
            G2VLM_AVAILABLE = False
            #if g2vlm_model is not None:
            raise ImportError("G2VLM Model Path pretrained need modifyed")

        if G2VLM_AVAILABLE:
            # Use G2VLM adapter
            logging.info("Initializing OmniVLA with G2VLM...")
            self.g2vlm_with_expert = G2VLMWithActorExpertModel(
                g2_vlm_path=g2_config_path,
                pretrained_g2vlm=self.use_pre_g2vlm,
                action_expert_config=action_expert_config,
                device=device,
            )
            logging.info("Using G2VLM adapter for PI-0 VLA")
        else:
            # Use PaliGemma (original)
            logging.info("Using PaliGemma adapter for PI-0 VLA")


        device = self.g2vlm_with_expert.g2vlm.device

        # need modify

        for param in self.g2vlm_with_expert.g2vlm.dino_model.parameters():
            param.requires_grad = False

        for param in self.g2vlm_with_expert.g2vlm.vit_model.parameters():
            param.requires_grad = False

        for param in self.g2vlm_with_expert.g2vlm.dino_model.parameters():
            param.requires_grad = False

        for param in self.g2vlm_with_expert.g2vlm.language_model.parameters():
            param.requires_grad = False

        for param in self.g2vlm_with_expert.g2vlm.point_decoder.parameters():
            param.requires_grad = False
        
        for param in self.g2vlm_with_expert.g2vlm.camera_decoder.parameters():
            param.requires_grad = False

        for param in self.g2vlm_with_expert.g2vlm.global_points_decoder.parameters():
            param.requires_grad = False

        hidden_size = self.g2vlm_with_expert.llm_config.hidden_size

        self.action_in_proj = nn.Linear(32, hidden_size).to(device = device)
        self.action_out_proj = nn.Linear(hidden_size, 32).to(device = device)

        self.state_proj = nn.Linear(32, hidden_size).to(device = device)
        self.action_time_mlp_in = nn.Linear(2 * hidden_size, hidden_size).to(device = device)
        self.action_time_mlp_out = nn.Linear(hidden_size, hidden_size).to(device = device)

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        # # Compile model if requested
        # if config.compile_model:
        #     torch.set_float32_matmul_precision("high")
        #     self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
        #     # Also compile the main forward pass used during training
        #     self.forward = torch.compile(self.forward, mode=config.compile_mode)

        # msg = """An incorrect transformer version is used, please create an issue on https://github.com/huggingface/lerobot/issues"""

        # try:
        #     from transformers.models.siglip import check

        #     if not check.check_whether_transformers_replace_is_installed_correctly():
        #         raise ValueError(msg)
        # except ImportError:
        #     raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        if self.use_pre_g2vlm:
            # G2VLM uses different structure
            self.g2vlm_with_expert.g2vlm.language_model.gradient_checkpointing = True
            self.g2vlm_with_expert.g2vlm.vit_model.gradient_checkpointing = True
        # Also enable for DINO model if it exists
            self.g2vlm_with_expert.g2vlm.dino_model.gradient_checkpointing = True
        else:
            # PaliGemma structure
            logging.info("Paligemma structure")
        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        if self.use_pre_g2vlm:
            # G2VLM uses different structure
            self.g2vlm_with_expert.g2vlm.language_model.gradient_checkpointing = False
            self.g2vlm_with_expert.g2vlm.vit_model.gradient_checkpointing = False
            # Also disable for DINO model if it exists
            self.g2vlm_with_expert.g2vlm.dino_model.gradient_checkpointing = False
        else:
            # PaliGemma structure
            logging.info("PaliGemma structure")
        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def _rtc_enabled(self):
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

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
        
    def _preprocess_observation_g2vlm(self, observation, *, train=True):
        """Helper method to preprocess observation for G2VLM."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(
            1.5, 1.0, bsize, device
        )
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_vlm_context(self, images, lang_tokens):
        """调用 G2VLM 获取双流融合特征 [3, 4]"""
        # G2VLM 会自动路由图像 Token 到语义和几何专家路径，并执行共享自注意力
        outputs = self.g2vlm_with_expert.g2vlm(
            input_ids=lang_tokens,
            pixel_values=images,
            output_hidden_states=True
        )
        # 获取最后的隐藏状态作为动作头的 context
        # 这里包含了 G2VLM 预测的点云几何特征和语义特征
        return outputs.last_hidden_state

    def embed_prefix_old(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer."""
        embs = []
        pad_masks = []
        att_masks = []
        
        logging.debug("embed_prefix_old images: type=%s len=%s shape=%s", type(images), len(images), images[0].shape if images else None)


        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.g2vlm_with_expert.embed_image(img)

            img_emb_dict = self._apply_checkpoint(image_embed_func, img)
            # 旧方法需要合并 semantic 和 geometric tokens
            semantic_tokens = img_emb_dict["semantic"]
            geometric_tokens = img_emb_dict["geometric"]
            img_emb = torch.cat([semantic_tokens, geometric_tokens], dim=1)
            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.g2vlm_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        # lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)
        lang_emb = lang_embed_func(lang_tokens=lang_tokens)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks
    
    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
        embs = []
        pad_masks = []
        att_masks = []
        token_type_ids = []

        # Add 20250110: 
        self.current_vit_grid = []  # 新增：用于存储 ViT 的 grid_thw
        self.current_dino_grid = [] # 新增：用于存储 DINO 的 grid_thw

        batch_size = lang_tokens.size(0)  # 以语言 token batch 为标准

        # Process images
        for img, img_mask in zip(images, img_masks):
            def image_embed_func(img):
                return self.g2vlm_with_expert.embed_image(img)
            img_emb_dict = self._apply_checkpoint(image_embed_func, img)
            semantic_tokens = img_emb_dict["semantic"]
            geometric_tokens = img_emb_dict["geometric"]

            # Add 20250110: 存储 grid 信息到 omni_vla 和 g2vlm_with_expert
            self.current_vit_grid.append(img_emb_dict["vit_grid"]) 
            self.current_dino_grid.append(img_emb_dict["dino_grid"])
            # 同时存储到 g2vlm_with_expert，以便在 forward 中使用
            if not hasattr(self.g2vlm_with_expert, 'current_vit_grid'):
                self.g2vlm_with_expert.current_vit_grid = []
                self.g2vlm_with_expert.current_dino_grid = []
            self.g2vlm_with_expert.current_vit_grid.append(img_emb_dict["vit_grid"])
            self.g2vlm_with_expert.current_dino_grid.append(img_emb_dict["dino_grid"])

            logging.debug("embed_prefix semantic max=%s geometric max=%s", semantic_tokens.abs().max().item(), geometric_tokens.abs().max().item())

            # expand batch to match language tokens batch
            if semantic_tokens.size(0) != batch_size:
                semantic_tokens = semantic_tokens.expand(batch_size, -1, -1)
            if geometric_tokens.size(0) != batch_size:
                geometric_tokens = geometric_tokens.expand(batch_size, -1, -1)

            embs.extend([semantic_tokens, geometric_tokens])
            pad_masks.extend([
                img_mask[:, None].expand(semantic_tokens.size(0), semantic_tokens.size(1)),
                img_mask[:, None].expand(geometric_tokens.size(0), geometric_tokens.size(1))
            ])
            att_masks.extend([0] * (semantic_tokens.size(1) + geometric_tokens.size(1)))
            # token type
            token_type_ids.extend([0] * semantic_tokens.size(1))  # semantic
            token_type_ids.extend([1] * geometric_tokens.size(1)) # geometric


        # Process language tokens
        lang_emb = self.g2vlm_with_expert.embed_language_tokens(lang_tokens)
        lang_emb = lang_emb * math.sqrt(lang_emb.size(-1))
        logging.debug("embed_prefix lang_emb max=%s", lang_emb.abs().max().item())

        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att_masks.extend([0] * lang_emb.size(1))
        token_type_ids.extend([2] * lang_emb.size(1))

        # concat along token dimension
        prefix_embeds = torch.cat(embs, dim=1)           # [B, N, D]
        prefix_pad_masks = torch.cat(pad_masks, dim=1)   # [B, N]
        prefix_att_masks = torch.tensor(att_masks, dtype=torch.bool, device=prefix_pad_masks.device)
        prefix_att_masks = prefix_att_masks[None, :].expand(batch_size, len(att_masks))
        prefix_token_type_ids = torch.tensor(token_type_ids, dtype=torch.long, device=prefix_pad_masks.device)
        prefix_token_type_ids = prefix_token_type_ids[None, :].expand(batch_size, len(token_type_ids))

        return prefix_embeds, prefix_pad_masks, prefix_att_masks, prefix_token_type_ids

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        device = self.state_proj.weight.device
        if state.dtype != self.state_proj.weight.dtype or state.device != device:
            state = state.to(dtype=self.state_proj.weight.dtype, device=device)

        def state_proj_func(state):
            return self.state_proj(state)

        state_emb = self._apply_checkpoint(state_proj_func, state)
        embs.append(state_emb[:, None, :])
        bsize = state_emb.shape[0]
        # device = state_emb.device

        state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)
        att_masks += [1]

        # Embed timestep using sine-cosine positional encoding
        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        if noisy_actions.dtype != self.action_in_proj.weight.dtype or noisy_actions.device != device:
            noisy_actions = noisy_actions.to(dtype=self.action_in_proj.weight.dtype, device=device)


        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)
        


        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if action_emb.device != device:
            action_emb = action_emb.to(device = device)
        

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        def mlp_func(action_time_emb):
            x = self.action_time_mlp_in(action_time_emb)
            x = F.silu(x)
            return self.action_time_mlp_out(x)

        action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
        adarms_cond = None

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

        return embs, pad_masks, att_masks, adarms_cond


    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        
        
        
        prefix_embs, prefix_pad_masks, prefix_att_masks, prefix_token_type_ids  = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        if (
            self.g2vlm_with_expert.g2vlm.language_model.model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        # Add 20250110
        # 收集构造 3D 索引所需的元数据
        prefix_info = {
            'batch_size': prefix_embs.shape[0],
            'device': prefix_embs.device,
            # 使用 torch.prod 计算 T*H*W，再 sum 起来
            'vit_len': sum([torch.prod(g).item() for g in self.current_vit_grid]), 
            'dino_len': sum([torch.prod(g).item() for g in self.current_dino_grid]),
            'text_len': lang_tokens.shape[1],
            'vit_grid': self.current_vit_grid,
            'dino_grid': self.current_dino_grid,
            'actual_prefix_len': prefix_embs.shape[1]  # 添加实际的 prefix 长度
        }

        logging.debug(
            "forward prefix_embs=%s suffix_embs=%s vit_grids=%s dino_grids=%s",
            prefix_embs.shape, suffix_embs.shape, len(self.current_vit_grid), len(self.current_dino_grid),
        )
        position_ids = self.build_3d_position_ids(prefix_info, suffix_embs.shape[1])

        # --- 🚀 核心修改点：混合因果掩码 ---
        # 替换原有的 make_att_2d_masks，注入 PI-0 的因果逻辑
        att_2d_masks_4d = self.build_pi0_attention_mask(prefix_pad_masks, suffix_embs.shape[1])

        # pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        # att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        # att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        # position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # # Prepare attention masks
        # att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Apply gradient checkpointing if enabled
        def forward_func(observation, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.g2vlm_with_expert.forward(
                observation = observation,
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, observation, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # Apply gradient checkpointing to final action projection if enabled
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks, prefix_token_type_ids = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)

        # G2VLM 的 cache 与 Gemma 不兼容，推理时每步用 Case 3 联合 forward(prefix_embs, suffix_embs)，不依赖 prefill cache
        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                prefix_embs,
                x_t,
                expanded_time,
            )

            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        prefix_embs,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep.
        使用 Case 3 联合 forward(prefix_embs, suffix_embs)，position_ids 由 forward 内根据 current_vit_grid/dino_grid 构建。
        """
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]

        if (
            self.g2vlm_with_expert.g2vlm.language_model.model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        # 与训练一致：用 build_pi0_attention_mask 构造 prefix+suffix 的因果掩码
        att_2d_masks_4d = self.build_pi0_attention_mask(prefix_pad_masks, suffix_len)
        # position_ids=None 时 G2VLM forward Case 3 会按 current_vit_grid/current_dino_grid 构建 3D position_ids
        if not self.use_pre_g2vlm:
            self.g2vlm_with_expert.action_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.g2vlm_with_expert.forward(
            observation=None,
            attention_mask=att_2d_masks_4d,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)
    
    # Add 20250110
    # 把这些分散的 grid_thw 拼成一个完整的 3D 坐标系
    def build_3d_position_ids(self, prefix_info, suffix_len):
        b = prefix_info['batch_size']
        device = prefix_info['device']
        curr_pos_val = 0  # 建议换个变量名，避免混淆

        # 1. 语义索引 (ViT)
        all_vit_pos = []
        for grid in prefix_info['vit_grid']:
            # 🚀 修复点：将 curr_pos 改为 curr_position_id
            # pos_3d, delta = get_rope_index_image_3D(
            #     grid.flatten(), 
            #     curr_position_id=curr_pos_val 
            # )
            pos_3d, delta = get_rope_index_image_3D(grid.flatten()[:3], curr_position_id=curr_pos_val)
            all_vit_pos.append(pos_3d.unsqueeze(1).repeat(1, b, 1))
            curr_pos_val += int(delta) + 1

        # 2. 几何索引 (DINO)
        all_dino_pos = []
        for grid in prefix_info['dino_grid']:
            # 🚀 同样修复这里的参数名
            # pos_3d, delta = get_rope_index_image_3D_dino(
            #     grid.flatten(), 
            #     curr_position_id=curr_pos_val
            # )
            pos_3d, delta = get_rope_index_image_3D(grid.flatten()[:3], curr_position_id=curr_pos_val)
            all_dino_pos.append(pos_3d.unsqueeze(1).repeat(1, b, 1))
            curr_pos_val += int(delta) + 1

        # # 3. 文本与动作 (线性 T 轴)
        # text_act_len = prefix_info['text_len'] + suffix_len
        # # 从最后的 curr_pos_val 开始递增
        # incremental_ids = torch.arange(curr_pos_val, curr_pos_val + text_act_len, device=device)
        
        # t_axis = incremental_ids.unsqueeze(0).repeat(b, 1)
        # h_axis = torch.zeros_like(t_axis)
        # w_axis = torch.zeros_like(t_axis)
        # text_act_pos = torch.stack([t_axis, h_axis, w_axis], dim=0)

        # # 4. 拼接全序列
        # full_vit_pos = torch.cat(all_vit_pos, dim=-1)
        # full_dino_pos = torch.cat(all_dino_pos, dim=-1)

        # 3. 文本索引
        # 💡 这里要动态计算文本的真实长度，防止 text_len 不准
        # 总 prefix 长度 - 已分配的视觉长度 = 文本长度
        current_vision_len = sum([p.shape[-1] for p in all_vit_pos]) + sum([p.shape[-1] for p in all_dino_pos])
        # 从 prefix_info 中获取实际的 prefix 长度，如果没有则从 text_len 和视觉长度计算
        actual_prefix_len = prefix_info.get('actual_prefix_len', current_vision_len + prefix_info.get('text_len', 0))
        text_len = actual_prefix_len - current_vision_len
        
        # 4. 拼接文本和动作 (Suffix)
        total_incremental_len = text_len + suffix_len

        if total_incremental_len <= 0:
            import os
            print(f"[Rank {os.environ.get('RANK')}] ERROR: total_incremental_len is {total_incremental_len}! curr_pos_val: {curr_pos_val}")
            # 临时给个 0 长度防止崩溃，观察是否能继续跑出更多 log
            total_incremental_len = 0

        incremental_ids = torch.arange(curr_pos_val, curr_pos_val + total_incremental_len, device=device)
        text_act_pos = incremental_ids.unsqueeze(0).unsqueeze(0).repeat(3, b, 1)

        # 5. 拼接全序列
        full_pos = torch.cat(all_vit_pos + all_dino_pos + [text_act_pos], dim=-1)

        # 🚨 最终断言：如果还是不等于 2403，说明拼接逻辑有根本性误解
        assert full_pos.shape[-1] == (actual_prefix_len + suffix_len), \
            f"Length Mismatch: PosIDs {full_pos.shape[-1]} != Embeds {actual_prefix_len + suffix_len}"
            
        return full_pos.to(device)
        
        # return torch.cat([full_vit_pos, full_dino_pos, text_act_pos], dim=-1).to(device)

    # Add 20250110
    # 把 prefix_pad_masks 扩展，并注入因果性
    def build_pi0_attention_mask(self, prefix_pad_masks, suffix_len):
        """
        prefix_pad_masks: [B, prefix_L] (你 embed_prefix 返回的那个)
        suffix_len: 动作长度
        """
        b, prefix_len = prefix_pad_masks.shape
        total_len = prefix_len + suffix_len
        device = prefix_pad_masks.device

        # 1. 构造 2D 基础掩码 [B, total_L, total_L]
        # 先初始化为全 True (可见)
        mask_2d = torch.ones((b, total_len, total_len), dtype=torch.bool, device=device)

        # 2. 处理 Padding (视觉/文本可能存在补齐)
        # 让所有 Token 遵守 prefix 的 padding 规则
        prefix_mask_expanded = prefix_pad_masks.unsqueeze(1).expand(-1, total_len, -1)
        mask_2d[:, :, :prefix_len] &= prefix_mask_expanded

        # 3. 注入因果律 (动作不能看未来)
        # 仅对 Suffix 区域应用下三角掩码
        causal_mask = torch.tril(torch.ones((suffix_len, suffix_len), device=device, dtype=torch.bool))
        # 动作区（右下角）
        mask_2d[:, prefix_len:, prefix_len:] &= causal_mask

        # 4. ❗VLA 关键：prefix 不能看 action
        mask_2d[:, :prefix_len, prefix_len:] = False

        # mask = mask_2d[0].int()   # 取第一个 batch，True/False → 1/0
        # print(mask)

        # plt.figure(figsize=(6, 6))
        # plt.imshow(mask_2d[0].cpu(), cmap="gray")
        # plt.axvline(prefix_len - 0.5, color="red")
        # plt.axhline(prefix_len - 0.5, color="red")
        # plt.title("VLA Attention Mask")

        # plt.savefig("vla_attention_mask.png", dpi=200, bbox_inches="tight")
        # plt.close()   # 很重要，防止显存/句柄泄露


        # 4. 映射为数值掩码 (-inf)
        return self._prepare_attention_masks_4d(mask_2d)
