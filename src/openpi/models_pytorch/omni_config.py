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
    g2vlm_path: str = ""  # checkpoint 或 HF repo

    # ------------------------------------------------------------------
    # 基础模型配置（覆盖 Pi0）
    # ------------------------------------------------------------------
    dtype: str = "bfloat16"
    action_expert_variant: str = "gemma_300m"


    use_pretrained_g2vlm: bool = False
    pretrained_g2vlm_path : str = '/data/openpi_temp/checkpoints/pi0_libero_low_mem_finetune/omni_9/30000'  # checkpoint 或 HF repo
    g2vlm_config_path : str = "/home/user/robot/model/G2VLM-2B-MoT"  # checkpoint 或 HF repo

    action_dim: int = 32
    action_horizon: int = 50

    # PI0 / PI05 行为
    pi05: bool = False

    # ------------------------------------------------------------------
    # 训练 / 冻结策略
    # ------------------------------------------------------------------
    # 这些是「参数名路径前缀」，供 PyTorch 使用
    frozen_patterns: tuple[str, ...] = (
        "g2vlm.dino_model",
        "g2vlm.vit_model",
        "g2vlm.language_model",
        "g2vlm.point_decoder",
        "g2vlm.camera_decoder",
        "g2vlm.global_points_decoder",
    )

    # JAX / 自动化脚本参考
    use_lora: bool = False

    # ------------------------------------------------------------------
    # 自动推导 token 长度
    # ------------------------------------------------------------------
    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(
                self,
                "max_token_len",
                256 if self.pi05 else 128,
            )

    # ------------------------------------------------------------------
    # Model type（⚠️ 不能复用 PI0 / PI05）
    # ------------------------------------------------------------------
    @property
    @override
    def model_type(self) -> _model.ModelType:
        # 如果你愿意，后续可以在 ModelType 里正式加一个 OMNI
        return "OMNI_VLA"  # PyTorch 路径不会用 enum

    # ------------------------------------------------------------------
    # PyTorch 实例化入口（核心）
    # ------------------------------------------------------------------
    def get_pytorch_model(
        self,
        g2vlm_instance_path: Optional[str] = None,
    ):
        """
        用于 serve_policy / train_policy
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
        return _model.ModelType.PI0  # 或 PI05

    # ------------------------------------------------------------------
    # 冻结规则（PyTorch 友好）
    # ------------------------------------------------------------------
    @override
    def get_freeze_filter(self) -> list[str]:
        """
        返回需要被冻结的参数路径前缀。
        PyTorch 脚本可直接：
            if name.startswith(prefix): param.requires_grad = False
        """
        return list(self.frozen_patterns)
