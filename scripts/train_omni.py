"""
PyTorch training entrypoint for PI0/PI05 with multi-GPU and multi-node (DDP) support.
This script mirrors the behavior of the JAX trainer (`scripts/train.py`) but runs
entirely in PyTorch using the `PI0Pytorch` model and your existing config/data
pipeline from `src/openpi/training/config.py` and `src/openpi/training/data_loader.py`.

Usage
Single GPU:
  python scripts/train_pytorch.py <config_name> --exp_name <run_name> --save_interval <interval>
  Example:
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test --resume  # Resume from latest checkpoint
Multi-GPU (single node):
  torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>
  Example:
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test --resume
Multi-Node Training:
	torchrun \
    --nnodes=<num_nodes> --nproc_per_node=<gpus_per_node> --node_rank=<rank_of_node> \
    --master_addr=<master_ip> --master_port=<port> \
    scripts/train_pytorch.py <config_name> --exp_name=<run_name> --save_interval <interval>

"""

import dataclasses
import gc
import logging
import os
import platform
import shutil
import time

import jax
import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.parallel
import tqdm
import wandb

import openpi.models.pi0_config
import openpi.models_pytorch.g2vlm_pi0_pytorch
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data
from openpi.vlm_expert.dinov2_with_registers.configuration_dinov2_with_registers import Dinov2WithRegistersConfig
from openpi.vlm_expert.dinov2_with_registers.modeling_dinov2_with_registers import Dinov2WithRegistersModel
from openpi.vlm_expert.g2vlm.g2vlm import G2VLM
from openpi.vlm_expert.g2vlm.g2vlm import G2VLMConfig
from openpi.vlm_expert.g2vlm.qwen2vl import Qwen2VLForCausalLM
from openpi.vlm_expert.qwen2.tokenization_qwen2 import Qwen2Tokenizer
from openpi.vlm_expert.qwen2vl.configuration_qwen2_vl import Qwen2VLConfig
from openpi.vlm_expert.qwen2vl.configuration_qwen2_vl import Qwen2VLVisionConfig
from openpi.vlm_expert.qwen2vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel
from openpi.models_pytorch.omni_vla import OmniVLA

from openpi.data_vlm.data_utils import add_special_tokens
from openpi.data_vlm.transforms import QwenVL2ImageTransform
from openpi.data_vlm.transforms import ImageTransform, InternVLImageTransform, QwenVL2ImageTransform
from openpi.data_vlm.transforms_vggt import DinoImageTransform, DinoImageNormalizeTransform
from safetensors.torch import load_file

from openpi.vlm_expert.qwen2vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel
from openpi.vlm_expert.g2vlm.qwen2vl import Qwen2VLForCausalLM
from openpi.vlm_expert.qwen2vl.configuration_qwen2_vl import Qwen2VLVisionConfig




# Load G2VLM config from checkpoint directory
def load_g2vlm_config_from_checkpoint(model_path: str) -> G2VLMConfig:
    """
    Load G2VLM configuration from checkpoint directory.
    
    Args:
        model_path: Path to the G2VLM checkpoint directory containing:
                   - text_config.json (for Qwen2VLConfig)
                   - vit_config.json (for Qwen2VLVisionConfig)
                   - dino_config.json (for Dinov2WithRegistersConfig)
    
    Returns:
        G2VLMConfig object initialized from the config files
    """
    import json
    
    llm_config_path = os.path.join(model_path, "text_config.json")
    vit_config_path = os.path.join(model_path, "vit_config.json")
    dino_config_path = os.path.join(model_path, "dino_config.json")
    
    # Load LLM config
    if os.path.exists(llm_config_path):
        # Try from_json_file first, fallback to from_dict if not available
        try:
            llm_config = Qwen2VLConfig.from_json_file(llm_config_path)
        except AttributeError:
            # Fallback to loading JSON and using from_dict
            with open(llm_config_path, 'r') as f:
                llm_config_dict = json.load(f)
            llm_config = Qwen2VLConfig.from_dict(llm_config_dict)
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = 'Qwen2VLMoTDecoderLayer'
        logging.info(f"Loaded LLM config from {llm_config_path}")
    else:
        logging.warning(f"LLM config not found at {llm_config_path}, using default config")
        llm_config = Qwen2VLConfig()
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = 'Qwen2VLMoTDecoderLayer'
    
    # Load VIT config
    if os.path.exists(vit_config_path):
        try:
            vit_config = Qwen2VLVisionConfig.from_json_file(vit_config_path)
        except AttributeError:
            with open(vit_config_path, 'r') as f:
                vit_config_dict = json.load(f)
            vit_config = Qwen2VLVisionConfig.from_dict(vit_config_dict)
        vit_config.patch_size = 14
        logging.info(f"Loaded VIT config from {vit_config_path}")
    else:
        logging.warning(f"VIT config not found at {vit_config_path}, using default config")
        vit_config = Qwen2VLVisionConfig()
        vit_config.patch_size = 14
    
    # Load DINO config
    if os.path.exists(dino_config_path):
        try:
            dino_config = Dinov2WithRegistersConfig.from_json_file(dino_config_path)
        except AttributeError:
            with open(dino_config_path, 'r') as f:
                dino_config_dict = json.load(f)
            dino_config = Dinov2WithRegistersConfig.from_dict(dino_config_dict)
        logging.info(f"Loaded DINO config from {dino_config_path}")
    else:
        logging.warning(f"DINO config not found at {dino_config_path}, using default config")
        dino_config = Dinov2WithRegistersConfig()
    
    # Create G2VLM config
    g2vlm_config = G2VLMConfig(
        visual_und=True,
        visual_recon=True,
        llm_config=llm_config,
        vit_config=vit_config,
        dino_config=dino_config,
        vit_max_num_patch_per_side=36,
    )
    
    return g2vlm_config


# add local model weight
def load_g2vlm_weights_from_checkpoint(model, model_path: str, device: torch.device):
    """
    Load pretrained G2VLM weights from a checkpoint directory.
    
    This function loads the G2VLM model weights from a safetensors file.
    The model structure should already be initialized in OMNIPytorch.
    
    Args:
        model: The OMNIPytorch model (may be wrapped in DDP)
        model_path: Path to the G2VLM checkpoint directory containing model.safetensors
        device: Target device
    """
    model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    g2vlm_model = model_to_load.g2vlm_with_expert.g2vlm
    
    model_state_dict_path = os.path.join(model_path, "model.safetensors")
    
    if not os.path.exists(model_state_dict_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_state_dict_path}")
    
    # Load model state dict
    logging.info(f"Loading G2VLM weights from {model_state_dict_path}")
    model_state_dict = safetensors.torch.load_file(model_state_dict_path, device="cpu")
    
    # Load weights into G2VLM model
    # The state dict should match the G2VLM model structure
    # Use strict=False to allow for partial loading (e.g., if action_expert weights are not in the checkpoint)
    msg = g2vlm_model.load_state_dict(model_state_dict, strict=False)
    
    logging.info(
        f"Loaded G2VLM weights. Missing keys: {len(msg.missing_keys)}, Unexpected keys: {len(msg.unexpected_keys)}"
    )
    
    if msg.missing_keys:
        logging.warning(f"Missing keys (first 20): {msg.missing_keys[:20]}")
        if len(msg.missing_keys) > 20:
            logging.warning(f"... and {len(msg.missing_keys) - 20} more missing keys")
    if msg.unexpected_keys:
        logging.warning(f"Unexpected keys (first 20): {msg.unexpected_keys[:20]}")
        if len(msg.unexpected_keys) > 20:
            logging.warning(f"... and {len(msg.unexpected_keys) - 20} more unexpected keys")
    
    # Clean up
    del model_state_dict
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logging.info(f"Successfully loaded G2VLM pretrained weights from {model_path}")

# add for wandb image logging
def normalize_to_hwc(img: torch.Tensor, cam_name: str):
    """
    支持:
      [C, H, W]
      [H, W, C]
    返回:
      [H, W, 3]
    """
    assert img.ndim == 3, f"{cam_name} ndim={img.ndim}"

    if img.shape[0] in (1, 3):
        # CHW -> HWC
        img = img.permute(1, 2, 0)
    elif img.shape[-1] in (1, 3):
        # 已经是 HWC
        pass
    else:
        raise ValueError(
            f"{cam_name} unknown image shape {tuple(img.shape)}"
        )

    # 保证 3 通道
    if img.shape[-1] == 1:
        img = img.repeat(1, 1, 3)
    elif img.shape[-1] > 3:
        img = img[..., :3]

    return img


def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    """Initialize wandb logging."""
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    if use_ddp and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")

        # Set up debugging environment variables for DDP issues
        if os.environ.get("TORCH_DISTRIBUTED_DEBUG") is None:
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    return use_ddp, local_rank, device


def cleanup_ddp():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def set_seed(seed: int, local_rank: int):
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + local_rank)


def build_datasets(config: _config.TrainConfig):
    # Use the unified data loader with PyTorch framework
    data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=True)
    return data_loader, data_loader.data_config()


def get_model_state_dict(model):
    """Get state dict from model, handling DDP wrapper."""
    return (
        model.module.state_dict()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.state_dict()
    )


def get_model_parameters(model):
    """Get parameters from model, handling DDP wrapper."""
    return (
        model.module.parameters()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.parameters()
    )


def save_checkpoint(model, optimizer, global_step, config, is_main, data_config):
    """Save a checkpoint with model state, optimizer state, and metadata."""
    if not is_main:
        return

    # Only save if it's time to save or if it's the final step
    if (global_step % config.save_interval == 0 and global_step > 0) or global_step == config.num_train_steps - 1:
        # Create temporary directory for atomic checkpoint saving
        final_ckpt_dir = config.checkpoint_dir / f"{global_step}"
        tmp_ckpt_dir = config.checkpoint_dir / f"tmp_{global_step}"

        # Remove any existing temp directory and create new one
        if tmp_ckpt_dir.exists():
            shutil.rmtree(tmp_ckpt_dir)
        tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save model state using safetensors (handle shared tensors)
        model_to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        safetensors.torch.save_model(model_to_save, tmp_ckpt_dir / "model.safetensors")

        # Save optimizer state using PyTorch format
        torch.save(optimizer.state_dict(), tmp_ckpt_dir / "optimizer.pt")

        # Save training metadata (avoid saving full config to prevent JAX/Flax compatibility issues)
        metadata = {
            "global_step": global_step,
            "config": dataclasses.asdict(config),
            "timestamp": time.time(),
        }
        torch.save(metadata, tmp_ckpt_dir / "metadata.pt")

        # save norm stats
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(tmp_ckpt_dir / "assets" / data_config.asset_id, norm_stats)

        # Atomically move temp directory to final location
        if final_ckpt_dir.exists():
            shutil.rmtree(final_ckpt_dir)
        tmp_ckpt_dir.rename(final_ckpt_dir)

        logging.info(f"Saved checkpoint at step {global_step} -> {final_ckpt_dir}")

        # Log checkpoint to wandb
        if config.wandb_enabled:
            wandb.log({"checkpoint_step": global_step}, step=global_step)


def load_checkpoint(model, optimizer, checkpoint_dir, device):
    """Load the latest checkpoint and return the global step."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]

    if not checkpoint_steps:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    latest_step = max(checkpoint_steps)
    ckpt_dir = checkpoint_dir / f"{latest_step}"

    # Clear memory before loading checkpoints
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "before_loading_checkpoint")

    try:
        # Load model state with error handling
        logging.info("Loading model state...")
        safetensors_path = ckpt_dir / "model.safetensors"

        if safetensors_path.exists():
            model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            safetensors.torch.load_model(model_to_load, safetensors_path, device=str(device))
            logging.info("Loaded model state from safetensors format")
        else:
            raise FileNotFoundError(f"No model checkpoint found at {ckpt_dir}")

        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_model")

        # Load optimizer state with error handling
        logging.info("Loading optimizer state...")
        optimizer_path = ckpt_dir / "optimizer.pt"

        if optimizer_path.exists():
            optimizer_state_dict = torch.load(optimizer_path, map_location=device, weights_only=False)
            logging.info("Loaded optimizer state from pt format")
        else:
            raise FileNotFoundError(f"No optimizer checkpoint found at {ckpt_dir}")

        optimizer.load_state_dict(optimizer_state_dict)
        del optimizer_state_dict
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_optimizer")

        # Load metadata
        logging.info("Loading metadata...")
        metadata = torch.load(ckpt_dir / "metadata.pt", map_location=device, weights_only=False)
        global_step = metadata.get("global_step", latest_step)
        del metadata
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_metadata")

        logging.info(f"Successfully loaded all checkpoint components from step {latest_step}")
        return global_step

    except RuntimeError as e:
        if "out of memory" in str(e):
            # Clear memory and provide detailed error message
            torch.cuda.empty_cache()
            gc.collect()
            logging.error(f"Out of memory error while loading checkpoint: {e!s}")
            log_memory_usage(device, latest_step, "after_oom_error")
            raise RuntimeError(
                "Out of memory while loading checkpoint. Try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
            ) from e
        raise


def get_latest_checkpoint_step(checkpoint_dir):
    """Get the latest checkpoint step number from a checkpoint directory."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    return max(checkpoint_steps) if checkpoint_steps else None


def log_memory_usage(device, step, phase="unknown"):
    """Log detailed memory usage information."""
    if not torch.cuda.is_available():
        return

    memory_allocated = torch.cuda.memory_allocated(device) / 1e9
    memory_reserved = torch.cuda.memory_reserved(device) / 1e9
    memory_free = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    memory_free = memory_free / 1e9

    # Get more detailed memory info
    memory_stats = torch.cuda.memory_stats(device)
    max_memory_allocated = memory_stats.get("allocated_bytes.all.peak", 0) / 1e9
    max_memory_reserved = memory_stats.get("reserved_bytes.all.peak", 0) / 1e9

    # Get DDP info if available
    ddp_info = ""
    if dist.is_initialized():
        ddp_info = f" | DDP: rank={dist.get_rank()}, world_size={dist.get_world_size()}"

    logging.info(
        f"Step {step} ({phase}): GPU memory - allocated: {memory_allocated:.2f}GB, reserved: {memory_reserved:.2f}GB, free: {memory_free:.2f}GB, peak_allocated: {max_memory_allocated:.2f}GB, peak_reserved: {max_memory_reserved:.2f}GB{ddp_info}"
    )


def train_loop(config: _config.TrainConfig):
    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    set_seed(config.seed, local_rank)

    # Initialize checkpoint directory and wandb
    resuming = False
    if config.resume:
        # Find checkpoint directory based on experiment name
        exp_checkpoint_dir = config.checkpoint_dir
        if exp_checkpoint_dir.exists():
            # Use validation to find the latest working checkpoint
            latest_step = get_latest_checkpoint_step(exp_checkpoint_dir)
            if latest_step is not None:
                resuming = True
                logging.info(
                    f"Resuming from experiment checkpoint directory: {exp_checkpoint_dir} at step {latest_step}"
                )
            else:
                raise FileNotFoundError(f"No valid checkpoints found in {exp_checkpoint_dir} for resume")
        else:
            raise FileNotFoundError(f"Experiment checkpoint directory {exp_checkpoint_dir} does not exist for resume")
    elif config.overwrite and config.checkpoint_dir.exists():
        shutil.rmtree(config.checkpoint_dir)
        logging.info(f"Overwriting checkpoint directory: {config.checkpoint_dir}")

    # Create checkpoint directory with experiment name
    if not resuming:
        # For new runs, create experiment-specific checkpoint directory
        exp_checkpoint_dir = config.checkpoint_dir
        exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created experiment checkpoint directory: {exp_checkpoint_dir}")
    else:
        # For resume, checkpoint_dir is already set to the experiment directory
        logging.info(f"Using existing experiment checkpoint directory: {config.checkpoint_dir}")

    # Initialize wandb (only on main process)
    if is_main:
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # Build data loader using the unified data loader
    # Calculate effective batch size per GPU for DDP
    # For N GPUs, each GPU should get batch_size/N samples, so total across all GPUs is batch_size
    world_size = torch.distributed.get_world_size() if use_ddp else 1
    effective_batch_size = config.batch_size // world_size
    logging.info(
        f"Using batch size per GPU: {effective_batch_size} (total batch size across {world_size} GPUs: {config.batch_size})"
    )

    # Pass the original batch size to data loader - it will handle DDP splitting internally
    loader, data_config = build_datasets(config)

    # Log sample images to wandb on first batch
    if is_main and config.wandb_enabled and not resuming:
        # Create a separate data loader for sample batch to avoid consuming the main loader
        sample_data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=False)
        sample_batch = next(iter(sample_data_loader))
        # Convert observation and actions to torch tensors
        observation, actions = sample_batch
        sample_batch = observation.to_dict()
        sample_batch["actions"] = actions

        # Create sample images for wandb
        images_to_log = []
        # Get batch size from the first image tensor
        image_dict = sample_batch["image"]
        batch_size = next(iter(sample_batch["image"].values())).shape[0]
        print(sample_batch["image"].keys(), np.shape)
        # Ori codeginial code had a bug here, fixed it, The Image shape problem 
        # for i in range(min(5, batch_size)):
        #     # Concatenate all camera views horizontally for this batch item
        #     # Convert from NCHW to NHWC format for wandb
        #     img_concatenated = torch.cat([img[i].permute(1, 2, 0) for img in sample_batch["image"].values()], axis=1)
        #     img_concatenated = img_concatenated.cpu().numpy()
        #     images_to_log.append(wandb.Image(img_concatenated))
        # here add by Yan
        for i in range(min(5, batch_size)):
            cam_imgs = []

            for cam_name in ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]:
                img = image_dict[cam_name]  # [B, H, W, C] 或 [B, C, H, W]
                single = img[i]

                single = normalize_to_hwc(single, cam_name)
                cam_imgs.append(single)

            img_concatenated = torch.cat(cam_imgs, dim=1)  # 拼宽度

            print("wandb image shape:", img_concatenated.shape)
            # 必须是 (224, 672, 3)

            images_to_log.append(
                wandb.Image(img_concatenated.cpu().numpy())
            )

        wandb.log({"camera_views": images_to_log}, step=0)

        # Clear sample batch from memory aggressively
        del sample_batch, observation, actions, images_to_log, img_concatenated
        del sample_data_loader  # Also delete the sample data loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("Cleared sample batch and data loader from memory")

    # Build model

    # if not isinstance(config.model, openpi.models.pi0_config.Pi0Config):
    #     # Convert dataclass to Pi0Config if needed
    #     model_cfg = openpi.models.pi0_config.Pi0Config(
    #         dtype=config.pytorch_training_precision,
    #         action_dim=config.model.action_dim,
    #         action_horizon=config.model.action_horizon,
    #         max_token_len=config.model.max_token_len,
    #         paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
    #         action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
    #         pi05=getattr(config.model, "pi05", False),
    #     )
    # else:
    #     model_cfg = config.model
    #     # Update dtype to match pytorch_training_precision
    #     object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)

    # model = openpi.models_pytorch.omni(OmniConfig=model_cfg).to(device)

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    model_cfg = config.model

    # Load G2VLM config from checkpoint if provided, otherwise use default
    if config.pytorch_weight_path is not None and os.path.exists(config.pytorch_weight_path):
        logging.info(f"Loading G2VLM config from checkpoint: {config.pytorch_weight_path}")
        vlm_config = load_g2vlm_config_from_checkpoint(config.pytorch_weight_path)
        # Store weight path in config for SpatialVLMWithExpertModel to use
        vlm_config.weight_path = config.pytorch_weight_path
    else:
        logging.info("Using default G2VLM config")
        vlm_config = G2VLMConfig()
        vlm_config.weight_path = None
    
    # model = openpi.models_pytorch.omni.OMNIPytorch(config=model_cfg, g2config=vlm_config).to(device)
    # g2_model, tokenizer, new_token_ids , vit_image_transform, dino_transform = load_model_and_tokenizer(config.pytorch_weight_path)
    g2_model_path = config.pytorch_weight_path
    g2_model_path = '/home/user/robot/model/G2VLM-2B-MoT'
    model = OmniVLA(config=model_cfg, device=device)

    model = model.to(device)

    enable_gradient_checkpointing = False
    model.gradient_checkpointing_disable()

    for n, p in model.named_parameters():
        if p.requires_grad:
            print(n)


    # if hasattr(model, "gradient_checkpointing_enable"):
    #     enable_gradient_checkpointing = True
    #     model.gradient_checkpointing_enable()
    #     logging.info("Enabled gradient checkpointing for memory optimization")
    # else:
    #     enable_gradient_checkpointing = False
    #     logging.info("Gradient checkpointing is not supported for this model")

    # Log initial memory usage after model creation
    if is_main and torch.cuda.is_available():
        log_memory_usage(device, 0, "after_model_creation")

    # Enable memory optimizations for large-scale training
    if world_size >= 8:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Set memory allocation configuration
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
        logging.info("Enabled memory optimizations for 8+ GPU training")

    if use_ddp:

        model = model.to(device)

        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,  # Disable for memory efficiency
            gradient_as_bucket_view=True,  # Enable for memory efficiency
            static_graph=world_size >= 8,  # Enable for 8+ GPUs
        )

    # Note: G2VLM weights are now loaded automatically in SpatialVLMWithExpertModel.__init__
    # if g2vlm_weight_path is provided in the config

    # Optimizer + learning rate schedule from config
    # omni_warmup_steps = 1_000
    # omni_peak_lr = 5e-5
    # omni_decay_steps = 1_000_000
    # omni_decay_lr = 5e-5

    # warmup_steps = omni_warmup_steps
    # peak_lr = omni_peak_lr
    # decay_steps = omni_decay_steps
    # end_lr = omni_decay_lr

    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr

    # Create optimizer with config parameters
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )


    # Load checkpoint if resuming
    global_step = 0
    if resuming:
        global_step = load_checkpoint(model, optim, config.checkpoint_dir, device)
        logging.info(f"Resumed training from step {global_step}")

    def lr_schedule(step: int):
        if step < warmup_steps:
            # Match JAX behavior: start from peak_lr / (warmup_steps + 1)
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        # cosine decay
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    model.train()

    print("Trainable parameters:")

    for n, p in model.named_parameters():
        if p.requires_grad:
            print(n)

    start_time = time.time()
    infos = []  # Collect stats over log interval
    if is_main:
        logging.info(
            f"Running on: {platform.node()} | world_size={torch.distributed.get_world_size() if use_ddp else 1}"
        )
        logging.info(
            f"Training config: batch_size={config.batch_size}, effective_batch_size={effective_batch_size}, num_train_steps={config.num_train_steps}"
        )
        logging.info(f"Memory optimizations: gradient_checkpointing={enable_gradient_checkpointing}")
        logging.info(
            f"LR schedule: warmup={warmup_steps}, peak_lr={peak_lr:.2e}, decay_steps={decay_steps}, end_lr={end_lr:.2e}"
        )
        logging.info(
            f"Optimizer: {type(config.optimizer).__name__}, weight_decay={config.optimizer.weight_decay}, clip_norm={config.optimizer.clip_gradient_norm}"
        )
        logging.info("EMA is not supported for PyTorch training")
        logging.info(f"Training precision: {model_cfg.dtype}")

    # Training loop - iterate until we reach num_train_steps
    pbar = (
        tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="Training", disable=not is_main)
        if is_main
        else None
    )

    # 打印一下看看，确保只有专家层和 Proj 层在训练
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Training: {name}")

    # Prepare inputs
    # 兼容单 GPU / DDP
    model_to_use = getattr(model, "module", model)

    # 总参数
    total_params = sum(p.numel() for p in model_to_use.parameters())
    trainable_params = sum(p.numel() for p in model_to_use.parameters() if p.requires_grad)

    print(f"\nTotal parameters: {total_params:,}  ({total_params / 1e9:.2f}B)")
    print(f"Trainable parameters: {trainable_params:,}  ({trainable_params / 1e9:.2f}B)")

    # 专家模块参数
    print(f"Reasoning params: {sum(p.numel() for p in model_to_use.reasoning_spatial_expert.reasoning_expert.parameters()) / 1e9:.2f}B")
    print(f"Spatial params: {sum(p.numel() for p in model_to_use.reasoning_spatial_expert.spatial_expert.parameters()) / 1e9:.2f}B")
    print(f"Action params: {sum(p.numel() for p in model_to_use.reasoning_spatial_expert.action_expert.parameters()) / 1e9:.2f}B")
    print(f"Spatial Encoder params: {sum(p.numel() for p in model_to_use.reasoning_spatial_expert.vggt_encoder.parameters()) / 1e9:.2f}B")


    while global_step < config.num_train_steps:
        # Set epoch for distributed training
        if use_ddp and hasattr(loader, "set_epoch"):
            loader.set_epoch(global_step // len(loader))

        for observation, actions in loader:
            # Check if we've reached the target number of steps
            if global_step >= config.num_train_steps:
                break

            optim.zero_grad(set_to_none=True)

            # The unified data loader returns (observation, actions) tuple
            # observation = jax.tree.map(lambda x: x.to(device), observation)  # noqa: PLW2901
            observation = jax.tree.map(lambda x: x.to(device).clone(), observation)
            actions = actions.to(torch.bfloat16)  # noqa: PLW2901
            actions = actions.to(device)  # noqa: PLW2901

            # Update LR
            for pg in optim.param_groups:
                pg["lr"] = lr_schedule(global_step)

            # Forward pass
            losses = model(observation, actions)
            # Ensure losses is a tensor and handle different return types
            if isinstance(losses, list | tuple):
                losses = torch.stack(losses)
            elif not isinstance(losses, torch.Tensor):
                losses = torch.tensor(losses, device=device, dtype=torch.float32)
            
            # --- 替换报错部分的 NaN 诊断代码 ---
            if torch.isnan(losses).any():
                print("\n" + "!"*50)
                print("检测到 Loss 为 NaN! 正在自动诊断...")
                
                # 1. 检查 Actions 输入
                if torch.isnan(actions).any():
                    print("错误：输入 actions 包含 NaN")
                
                # 2. 检查模型参数梯度（如果是回传后 NaN）
                for name, param in model.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        print(f"梯度异常：层 {name} 的梯度包含 NaN")
                
                # 3. 检查 Attention Mask
                # 建议在模型 forward 内部也打印一下 attention_mask.min()
                
                print("!"*50 + "\n")
                # 抛出异常停止程序，防止污染之后的 checkpoint
                raise RuntimeError("Stopping due to NaN loss")

            loss = losses.mean()

            # Backward pass
            loss.backward()

            

            # Log memory usage after backward pass
            if global_step < 5 and is_main and torch.cuda.is_available():
                log_memory_usage(device, global_step, "after_backward")

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.optimizer.clip_gradient_norm)
            torch.cuda.empty_cache()
            # Optimizer step
            optim.step()
            optim.zero_grad(set_to_none=True)

            # Clear gradients more aggressively
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad = None

            # Collect stats
            if is_main:
                infos.append(
                    {
                        "loss": loss.item(),
                        "learning_rate": optim.param_groups[0]["lr"],
                        "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    }
                )

            if is_main and (global_step % config.log_interval == 0):
                elapsed = time.time() - start_time

                # Average stats over log interval
                avg_loss = sum(info["loss"] for info in infos) / len(infos)
                avg_lr = sum(info["learning_rate"] for info in infos) / len(infos)

                avg_grad_norm = None
                if any("grad_norm" in info for info in infos):
                    vals = [
                        info["grad_norm"] for info in infos if "grad_norm" in info and info["grad_norm"] is not None
                    ]
                    if len(vals) > 0:
                        avg_grad_norm = sum(vals) / len(vals)
                logging.info(
                    f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} grad_norm={avg_grad_norm:.2f} time={elapsed:.1f}s"
                    if avg_grad_norm is not None
                    else f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} time={elapsed:.1f}s"
                )

                # Log to wandb
                if config.wandb_enabled and len(infos) > 0:
                    log_payload = {
                        "loss": avg_loss,
                        "learning_rate": avg_lr,
                        "step": global_step,
                        "time_per_step": elapsed / config.log_interval,
                    }
                    if avg_grad_norm is not None:
                        log_payload["grad_norm"] = avg_grad_norm
                    wandb.log(log_payload, step=global_step)

                start_time = time.time()
                infos = []  # Reset stats collection

            global_step += 1
            # Save checkpoint using the new mechanism
            save_checkpoint(model, optim, global_step, config, is_main, data_config)

            # Update progress bar
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "lr": f"{optim.param_groups[0]['lr']:.2e}", "step": global_step}
                )

    # Close progress bar
    if pbar is not None:
        pbar.close()

    # Finish wandb run
    if is_main and config.wandb_enabled:
        wandb.finish()

    cleanup_ddp()


def main():
    init_logging()
    config = _config.cli()
    train_loop(config)


if __name__ == "__main__":
    main()
