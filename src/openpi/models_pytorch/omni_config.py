import dataclasses
from typing import Optional, Literal, override
import torch
import openpi.models.pi0_config as _pi0_config
from openpi.models import gemma as _gemma

@dataclasses.dataclass(frozen=True)
class OmniConfig(_pi0_config.Pi0Config):
    # --- 新增 G2VLM 相关配置 ---
    g2vlm_path: str = ""  # G2VLM 模型权重路径
    
    # --- 覆盖或继承 PI0 的配置 ---
    dtype: str = "bfloat16"
    action_expert_variant: str = "gemma_300m" # 对应你代码中的 Action Expert
    
    # 机器人控制参数
    action_dim: int = 32
    action_horizon: int = 50
    
    # 模型模式
    pi05: bool = False  # 如果是 True，会影响 token 长度和 AdaRMS 使用
    
    # --- 训练/冻结逻辑适配 ---
    # 默认冻结 G2VLM 的所有视觉和语言组件，只训练 Action Expert 和投影层
    frozen_patterns: tuple[str, ...] = (
        "g2vlm.dino_model",
        "g2vlm.vit_model",
        "g2vlm.language_model",
        "g2vlm.point_decoder",
        "g2vlm.camera_decoder",
        "g2vlm.global_points_decoder",
    )

    # 这里的 freeze_filter 主要给 JAX 或 自动化脚本参考
    use_lora: bool = False 

    def __post_init__(self):
        # 自动设置 token 长度
        if self.max_token_len is None:
            # G2VLM 通常需要处理更多的视觉 token 或 3D token
            object.__setattr__(self, "max_token_len", 256 if self.pi05 else 128)
            
    @property
    def model_type(self):
        return "OmniVLA"

    def get_pytorch_model(self, g2vlm_instance_path: Optional[str] = None):
        """
        辅助方法：直接通过此 config 实例化你的 OmniVLA 模型
        """
        from .omni_vla import OmniVLA # 替换为 OmniVLA 所在的实际路径
        path = g2vlm_instance_path or self.g2vlm_path
        return OmniVLA(config=self, g2vlm_model=path)

    # 重写冻结逻辑以匹配你的代码需求
    @override
    def get_freeze_filter(self) -> list[str]:
        """
        返回在配置文件中定义的冻结模式。
        这个列表可以被 PyTorch 的训练脚本读取，用来设置 requires_grad。
        """
        return list(self.frozen_patterns)