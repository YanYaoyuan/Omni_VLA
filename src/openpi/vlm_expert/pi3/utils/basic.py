import os
import os.path as osp
import math
import cv2
from PIL import Image
import torch
from torchvision import transforms
from plyfile import PlyData, PlyElement
import numpy as np


import os
import random
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
import logging
from prettytable import PrettyTable
import itertools
from collections import defaultdict
from PIL import Image

def seed_anything(seed: int, deterministic=False):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:   # https://pytorch.org/docs/stable/notes/randomness.html
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        # torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.benchmark = False          ##############  if it is hard to select a best algorithm (to much resolution for convolution), this is much faster!

def colmap_to_opencv_intrinsics(K):
    """
    Modify camera intrinsics to follow a different convention.
    Coordinates of the center of the top-left pixels are by default:
    - (0.5, 0.5) in Colmap
    - (0,0) in OpenCV
    """
    K = K.copy()
    K[0, 2] -= 0.5
    K[1, 2] -= 0.5
    return K


def opencv_to_colmap_intrinsics(K):
    """
    Modify camera intrinsics to follow a different convention.
    Coordinates of the center of the top-left pixels are by default:
    - (0.5, 0.5) in Colmap
    - (0,0) in OpenCV
    """
    K = K.copy()
    K[0, 2] += 0.5
    K[1, 2] += 0.5
    return K

def is_logging_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def get_logger(cfg, name=None):
    # log_file_path is used when unit testing
    if is_logging_process():
        logging.config.dictConfig(
            OmegaConf.to_container(cfg.job_logging_cfg, resolve=True)
        )
        return logging.getLogger(name)
    

def count_parameters(model, top_k=15):
    table = PrettyTable([f"Modules (only show top {top_k} modules)", "Parameters"])
    total_params = 0
    param_count = defaultdict(int) 

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        top_level_module = name.split('.')[0]
        
        params = parameter.numel()
        param_count[top_level_module] += params
        
        total_params += params

    sorted_params = sorted(param_count.items(), key=lambda x: x[1], reverse=True)
    for i, (module_name, param_count) in enumerate(sorted_params):
        if i < top_k:
            table.add_row([module_name, param_count])

    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def tensor_to_pil(tensor):
    """
    Converts a PyTorch tensor to a PIL image. Automatically moves the channel dimension 
    (if it has size 3) to the last axis before converting.

    Args:
        tensor (torch.Tensor): Input tensor. Expected shape can be [C, H, W], [H, W, C], or [H, W].
    
    Returns:
        PIL.Image: The converted PIL image.
    """
    if torch.is_tensor(tensor):
        array = tensor.detach().cpu().numpy()
    else:
        array = tensor

    return array_to_pil(array)


def array_to_pil(array):
    """
    Converts a NumPy array to a PIL image. Automatically:
        - Squeezes dimensions of size 1.
        - Moves the channel dimension (if it has size 3) to the last axis.
    
    Args:
        array (np.ndarray): Input array. Expected shape can be [C, H, W], [H, W, C], or [H, W].
    
    Returns:
        PIL.Image: The converted PIL image.
    """
    # Remove singleton dimensions
    array = np.squeeze(array)
    
    # Ensure the array has the channel dimension as the last axis
    if array.ndim == 3 and array.shape[0] == 3:  # If the channel is the first axis
        array = np.transpose(array, (1, 2, 0))  # Move channel to the last axis
    
    # Handle single-channel grayscale images
    if array.ndim == 2:  # [H, W]
        return Image.fromarray((array * 255).astype(np.uint8), mode="L")
    elif array.ndim == 3 and array.shape[2] == 3:  # [H, W, C] with 3 channels
        return Image.fromarray((array * 255).astype(np.uint8), mode="RGB")
    else:
        raise ValueError(f"Unsupported array shape for PIL conversion: {array.shape}")


def find_best_alignment(gt_normal, pred_normal):
    """
    找出将 gt_normal 调整为 pred_normal 的最佳操作。
    
    参数：
    - gt_normal: Tensor of shape (N, 3)，GT 法向量。
    - pred_normal: Tensor of shape (N, 3)，预测的法向量。
    
    返回：
    - best_perm: 最佳的 xyz 轴顺序 (例如 (0, 1, 2))。
    - best_signs: 最佳的符号调整 (例如 [1, -1, 1])。
    - best_error: 调整后的最小误差。
    """
    # 所有可能的轴顺序排列
    permutations = list(itertools.permutations([0, 1, 2]))
    # 所有可能的符号组合 (+1, -1)
    signs = list(itertools.product([1, -1], repeat=3))

    best_error = float('inf')
    best_perm = None
    best_signs = None

    for perm in permutations:
        for sign in signs:
            # 调整 gt_normal 的顺序和符号
            transformed_gt = gt_normal[:, perm] * torch.tensor(sign, device=gt_normal.device)

            # 计算 L2 范数误差
            error = torch.norm(transformed_gt - pred_normal, dim=1).mean().item()

            # 更新最佳结果
            if error < best_error:
                best_error = error
                best_perm = perm
                best_signs = sign

    return best_perm, best_signs, best_error


def get_robust_pca(features: torch.Tensor, m: float = 2, remove_first_component=False):
    # features: (N, C)
    # m: a hyperparam controlling how many std dev outside for outliers
    assert len(features.shape) == 2, "features should be (N, C)"
    reduction_mat = torch.pca_lowrank(features, q=3, niter=20)[2]
    colors = features @ reduction_mat
    if remove_first_component:
        colors_min = colors.min(dim=0).values
        colors_max = colors.max(dim=0).values
        tmp_colors = (colors - colors_min) / (colors_max - colors_min)
        fg_mask = tmp_colors[..., 0] < 0.2
        reduction_mat = torch.pca_lowrank(features[fg_mask], q=3, niter=20)[2]
        colors = features @ reduction_mat
    else:
        fg_mask = torch.ones_like(colors[:, 0]).bool()
    d = torch.abs(colors[fg_mask] - torch.median(colors[fg_mask], dim=0).values)
    mdev = torch.median(d, dim=0).values
    s = d / mdev
    try:
        rins = colors[fg_mask][s[:, 0] < m, 0]
        gins = colors[fg_mask][s[:, 1] < m, 1]
        bins = colors[fg_mask][s[:, 2] < m, 2]
        rgb_min = torch.tensor([rins.min(), gins.min(), bins.min()])
        rgb_max = torch.tensor([rins.max(), gins.max(), bins.max()])
    except:
        rins = colors
        gins = colors
        bins = colors
        rgb_min = torch.tensor([rins.min(), gins.min(), bins.min()])
        rgb_max = torch.tensor([rins.max(), gins.max(), bins.max()])

    return reduction_mat, rgb_min.to(reduction_mat), rgb_max.to(reduction_mat)


def get_pca_map(
    feature_map: torch.Tensor,
    return_pca_stats=False,
    pca_stats=None,
):
    """
    feature_map: (1, h, w, C) is the feature map of a single image.
    """
    if feature_map.shape[0] != 1:
        # make it (1, h, w, C)
        feature_map = feature_map[None]
    if pca_stats is None:
        reduct_mat, color_min, color_max = get_robust_pca(
            feature_map.reshape(-1, feature_map.shape[-1])
        )
    else:
        reduct_mat, color_min, color_max = pca_stats
    pca_color = feature_map @ reduct_mat
    pca_color = (pca_color - color_min) / (color_max - color_min)
    pca_color = pca_color.clamp(0, 1)
    pca_color = pca_color.cpu().numpy().squeeze(0)
    if return_pca_stats:
        return pca_color, (reduct_mat, color_min, color_max)
    return pca_color


def load_images_as_tensor(path='data/truck', interval=1, PIXEL_LIMIT=255000):
    """
    Loads images from a directory or video, resizes them to a uniform size,
    then converts and stacks them into a single [N, 3, H, W] PyTorch tensor.
    """
    sources = [] 
    
    # --- 1. Load image paths or video frames ---
    if osp.isdir(path):
        print(f"Loading images from directory: {path}")
        filenames = sorted([x for x in os.listdir(path) if x.lower().endswith(('.png', '.jpg', '.jpeg'))])
        for i in range(0, len(filenames), interval):
            img_path = osp.join(path, filenames[i])
            try:
                sources.append(Image.open(img_path).convert('RGB'))
            except Exception as e:
                print(f"Could not load image {filenames[i]}: {e}")
    elif path.lower().endswith('.mp4'):
        print(f"Loading frames from video: {path}")
        cap = cv2.VideoCapture(path)
        if not cap.isOpened(): raise IOError(f"Cannot open video file: {path}")
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            if frame_idx % interval == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                sources.append(Image.fromarray(rgb_frame))
            frame_idx += 1
        cap.release()
    else:
        raise ValueError(f"Unsupported path. Must be a directory or a .mp4 file: {path}")

    if not sources:
        print("No images found or loaded.")
        return torch.empty(0)

    print(f"Found {len(sources)} images/frames. Processing...")

    # --- 2. Determine a uniform target size for all images based on the first image ---
    # This is necessary to ensure all tensors have the same dimensions for stacking.
    first_img = sources[0]
    W_orig, H_orig = first_img.size
    scale = math.sqrt(PIXEL_LIMIT / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1
    W_target, H_target = W_orig * scale, H_orig * scale
    k, m = round(W_target / 14), round(H_target / 14)
    while (k * 14) * (m * 14) > PIXEL_LIMIT:
        if k / m > W_target / H_target: k -= 1
        else: m -= 1
    TARGET_W, TARGET_H = max(1, k) * 14, max(1, m) * 14
    print(f"All images will be resized to a uniform size: ({TARGET_W}, {TARGET_H})")

    # --- 3. Resize images and convert them to tensors in the [0, 1] range ---
    tensor_list = []
    # Define a transform to convert a PIL Image to a CxHxW tensor and normalize to [0,1]
    to_tensor_transform = transforms.ToTensor()
    
    for img_pil in sources:
        try:
            # Resize to the uniform target size
            resized_img = img_pil.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
            # Convert to tensor
            img_tensor = to_tensor_transform(resized_img)
            tensor_list.append(img_tensor)
        except Exception as e:
            print(f"Error processing an image: {e}")

    if not tensor_list:
        print("No images were successfully processed.")
        return torch.empty(0)

    # --- 4. Stack the list of tensors into a single [N, C, H, W] batch tensor ---
    return torch.stack(tensor_list, dim=0)


def tensor_to_pil(tensor):
    """
    Converts a PyTorch tensor to a PIL image. Automatically moves the channel dimension 
    (if it has size 3) to the last axis before converting.

    Args:
        tensor (torch.Tensor): Input tensor. Expected shape can be [C, H, W], [H, W, C], or [H, W].
    
    Returns:
        PIL.Image: The converted PIL image.
    """
    if torch.is_tensor(tensor):
        array = tensor.detach().cpu().numpy()
    else:
        array = tensor

    return array_to_pil(array)


def array_to_pil(array):
    """
    Converts a NumPy array to a PIL image. Automatically:
        - Squeezes dimensions of size 1.
        - Moves the channel dimension (if it has size 3) to the last axis.
    
    Args:
        array (np.ndarray): Input array. Expected shape can be [C, H, W], [H, W, C], or [H, W].
    
    Returns:
        PIL.Image: The converted PIL image.
    """
    # Remove singleton dimensions
    array = np.squeeze(array)
    
    # Ensure the array has the channel dimension as the last axis
    if array.ndim == 3 and array.shape[0] == 3:  # If the channel is the first axis
        array = np.transpose(array, (1, 2, 0))  # Move channel to the last axis
    
    # Handle single-channel grayscale images
    if array.ndim == 2:  # [H, W]
        return Image.fromarray((array * 255).astype(np.uint8), mode="L")
    elif array.ndim == 3 and array.shape[2] == 3:  # [H, W, C] with 3 channels
        return Image.fromarray((array * 255).astype(np.uint8), mode="RGB")
    else:
        raise ValueError(f"Unsupported array shape for PIL conversion: {array.shape}")


def rotate_target_dim_to_last_axis(x, target_dim=3):
    shape = x.shape
    axis_to_move = -1
    # Iterate backwards to find the first occurrence from the end 
    # (which corresponds to the last dimension of size 3 in the original order).
    for i in range(len(shape) - 1, -1, -1):
        if shape[i] == target_dim:
            axis_to_move = i
            break

    # 2. If the axis is found and it's not already in the last position, move it.
    if axis_to_move != -1 and axis_to_move != len(shape) - 1:
        # Create the new dimension order.
        dims_order = list(range(len(shape)))
        dims_order.pop(axis_to_move)
        dims_order.append(axis_to_move)
        
        # Use permute to reorder the dimensions.
        ret = x.transpose(*dims_order)
    else:
        ret = x

    return ret


def write_ply(
    xyz,
    rgb=None,
    path='output.ply',
) -> None:
    if torch.is_tensor(xyz):
        xyz = xyz.detach().cpu().numpy()

    if torch.is_tensor(rgb):
        rgb = rgb.detach().cpu().numpy()

    if rgb is not None and rgb.max() > 1:
        rgb = rgb / 255.

    xyz = rotate_target_dim_to_last_axis(xyz, 3)
    xyz = xyz.reshape(-1, 3)

    if rgb is not None:
        rgb = rotate_target_dim_to_last_axis(rgb, 3)
        rgb = rgb.reshape(-1, 3)
    
    if rgb is None:
        min_coord = np.min(xyz, axis=0)
        max_coord = np.max(xyz, axis=0)
        normalized_coord = (xyz - min_coord) / (max_coord - min_coord + 1e-8)
        
        hue = 0.7 * normalized_coord[:,0] + 0.2 * normalized_coord[:,1] + 0.1 * normalized_coord[:,2]
        hsv = np.stack([hue, 0.9*np.ones_like(hue), 0.8*np.ones_like(hue)], axis=1)

        c = hsv[:,2:] * hsv[:,1:2]
        x = c * (1 - np.abs( (hsv[:,0:1]*6) % 2 - 1 ))
        m = hsv[:,2:] - c
        
        rgb = np.zeros_like(hsv)
        cond = (0 <= hsv[:,0]*6%6) & (hsv[:,0]*6%6 < 1)
        rgb[cond] = np.hstack([c[cond], x[cond], np.zeros_like(x[cond])])
        cond = (1 <= hsv[:,0]*6%6) & (hsv[:,0]*6%6 < 2)
        rgb[cond] = np.hstack([x[cond], c[cond], np.zeros_like(x[cond])])
        cond = (2 <= hsv[:,0]*6%6) & (hsv[:,0]*6%6 < 3)
        rgb[cond] = np.hstack([np.zeros_like(x[cond]), c[cond], x[cond]])
        cond = (3 <= hsv[:,0]*6%6) & (hsv[:,0]*6%6 < 4)
        rgb[cond] = np.hstack([np.zeros_like(x[cond]), x[cond], c[cond]])
        cond = (4 <= hsv[:,0]*6%6) & (hsv[:,0]*6%6 < 5)
        rgb[cond] = np.hstack([x[cond], np.zeros_like(x[cond]), c[cond]])
        cond = (5 <= hsv[:,0]*6%6) & (hsv[:,0]*6%6 < 6)
        rgb[cond] = np.hstack([c[cond], np.zeros_like(x[cond]), x[cond]])
        rgb = (rgb + m)

    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    normals = np.zeros_like(xyz)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb * 255), axis=1)
    elements[:] = list(map(tuple, attributes))
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(path)