import dataclasses
from typing import Optional
from typing_extensions import override

import openpi.models.pi0_config as _pi0_config
from openpi.models import model as _model


@dataclasses.dataclass(frozen=True)
class OmniConfig(_pi0_config.Pi0Config):
    """
    OmniVLA = G2VLM (frozen) + PI0 Action Expert
    """

    # ------------------------------------------------------------------
    # G2VLM
    # ------------------------------------------------------------------
    g2vlm_path: str = ""  # checkpoint or HF repo

    # ------------------------------------------------------------------
    # Base model config (overrides Pi0)
    # ------------------------------------------------------------------
    dtype: str = "bfloat16"
    action_expert_variant: str = "gemma_300m"


    use_pretrained_g2vlm: bool = False
    pretrained_g2vlm_path : str = '/data/openpi_temp/checkpoints/pi0_libero_low_mem_finetune/omni_9/30000'  # checkpoint or HF repo
    g2vlm_config_path : str = "/home/user/robot/model/G2VLM-2B-MoT"  # checkpoint or HF repo

    vlm_pretrained_path: Optional[str] = "/root/autodl-tmp/huggingface/lerobot/pi0_torch_libero/pi0_torch_libero/model.safetensors"
    vggt_pretrained_path: Optional[str] = "/root/autodl-tmp/huggingface/lerobot/VGGT-1B/model.safetensors"

    # ------------------------------------------------------------------
    # OmniVLA Pretrained Checkpoint (for fine-tuning)
    # ------------------------------------------------------------------
    omni_pretrained_path: Optional[str] = None

    action_dim: int = 32
    action_horizon: int = 50

    # PI0 / PI05 behavior
    pi05: bool = False

    freeze_vision_encoder : bool = True
    freeze_language_model : bool = True
    freeze_VGGT_model : bool = True
    train_expert_only : bool = False

    # ------------------------------------------------------------------
    # Training / Freezing Strategy
    # ------------------------------------------------------------------
    # These are parameter name prefixes for PyTorch
    frozen_patterns: tuple[str, ...] = (
        "g2vlm.dino_model",
        "g2vlm.vit_model",
        "g2vlm.language_model",
        "g2vlm.point_decoder",
        "g2vlm.camera_decoder",
        "g2vlm.global_points_decoder",
    )

    # JAX / Automation script reference
    use_lora: bool = False

    # ------------------------------------------------------------------
    # Automatically infer token length
    # ------------------------------------------------------------------
    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(
                self,
                "max_token_len",
                256 if self.pi05 else 128,
            )

    # ------------------------------------------------------------------
    # Model type（⚠️ cannot reuse PI0 / PI05）
    # ------------------------------------------------------------------
    @property
    @override
    def model_type(self) -> _model.ModelType:
        # If you want, you can officially add an OMNI in ModelType later
        return "OMNI_VLA"  # PyTorch path won't use enum

    # ------------------------------------------------------------------
    # PyTorch instantiation entrypoint (core)
    # ------------------------------------------------------------------
    def get_pytorch_model(
        self,
        g2vlm_instance_path: Optional[str] = None,
    ):
        """
        Used for serve_policy / train_policy
        """
        from openpi.models_pytorch.omni_vla import OmniVLA

        g2vlm_path = g2vlm_instance_path or self.g2vlm_path
        if not g2vlm_path:
            raise ValueError("OmniConfig requires g2vlm_path")

        return OmniVLA(
            config=self,
            g2vlm_model_path=g2vlm_path,
        )
    
    @property
    def model_type(self):
        return _model.ModelType.PI0  # or PI05

    # ------------------------------------------------------------------
    # Freeze rules (PyTorch friendly)
    # ------------------------------------------------------------------
    @override
    def get_freeze_filter(self) -> list[str]:
        """
        Returns parameter path prefixes to be frozen.
        PyTorch script can directly use:
            if name.startswith(prefix): param.requires_grad = False
        """
        return list(self.frozen_patterns)
