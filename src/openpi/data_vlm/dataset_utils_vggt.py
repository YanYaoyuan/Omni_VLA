# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import math
import numpy as np
from PIL import Image
import PIL
import re 
try:
    lanczos = PIL.Image.Resampling.LANCZOS
    bicubic = PIL.Image.Resampling.BICUBIC
except AttributeError:
    lanczos = PIL.Image.LANCZOS
    bicubic = PIL.Image.BICUBIC

import h5py
# from .frame_sampling_utils import 
from .frame_sampling_utils import compute_ranking

import torch.nn.functional as F
from petrel_client.client import Client
from io import BytesIO
import decimal
import quaternion
import json
import torch 
import torchvision
try:
    import open3d as o3d
except:
    print("Open3d is not installed, may install later for Visualization")
import os.path as osp
import pi3.utils.cropping as cropping
from pathlib import Path
from scipy.spatial.transform import Rotation
import imageio.v2 as iio


def transform_depth(depthmap, camera_intrinsic):
    H, W = depthmap.shape
    fx = camera_intrinsic[0, 0]
    fy = camera_intrinsic[1, 1]
    cx = camera_intrinsic[0, 2]
    cy = camera_intrinsic[1, 2]
    xs = np.arange(W)
    ys = np.arange(H)
    xx, yy = np.meshgrid(xs, ys)
    x = (xx - cx) / fx
    y = (yy - cy) / fy
    norm = np.sqrt(x**2 + y**2 + 1)
    depthmap = depthmap / norm
    return depthmap


def load_split_info(scene_dir: Path):
    """Return the split json dict."""
    with open(scene_dir / "split_info.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_camera_poses(scene_dir: Path, split_idx: int):
    """
    Returns
    -------
    intrinsics : (S, 3, 3) array, pixel-space K matrices
    extrinsics : (S, 4, 4) array, OpenCV world-to-camera matrices
    """
    # ----- read metadata -----------------------------------------------------
    split_info = load_split_info(scene_dir)
    frame_count = len(split_info["split"][split_idx])

    cam_file = scene_dir / "camera" / f"split_{split_idx}.json"
    with open(cam_file, "r", encoding="utf-8") as f:
        cam = json.load(f)

    # ----- intrinsics --------------------------------------------------------
    intrinsics = np.repeat(np.eye(3)[None, ...], frame_count, axis=0)
    intrinsics[:, 0, 0] = cam["focals"]          # fx
    intrinsics[:, 1, 1] = cam["focals"]          # fy
    intrinsics[:, 0, 2] = cam["cx"]              # cx
    intrinsics[:, 1, 2] = cam["cy"]              # cy

    # ----- extrinsics --------------------------------------------------------
    extrinsics = np.repeat(np.eye(4)[None, ...], frame_count, axis=0)

    # SciPy expects quaternions as (x, y, z, w) → convert
    quat_wxyz = np.array(cam["quats"])           # (S, 4)  (w,x,y,z)
    quat_xyzw = np.concatenate([quat_wxyz[:, 1:], quat_wxyz[:, :1]], axis=1)

    rotations = Rotation.from_quat(quat_xyzw).as_matrix()      # (S, 3, 3)
    translations = np.array(cam["trans"])               # (S, 3)

    extrinsics[:, :3, :3] = rotations
    extrinsics[:, :3, 3] = translations

    return intrinsics.astype(np.float32), extrinsics.astype(np.float32)



def crop_resize_if_necessary(self, image, depthmap, intrinsics, resolution, rng=None, info=None, normal=None, far_mask=None):
    """ This function:
        - first downsizes the image with LANCZOS inteprolation,
            which is better than bilinear interpolation in
    """
    if not isinstance(image, PIL.Image.Image):
        image = PIL.Image.fromarray(image)

    # downscale with lanczos interpolation so that image.size == resolution
    # cropping centered on the principal point
    W, H = image.size
    cx, cy = intrinsics[:2, 2].round().astype(int)
    min_margin_x = min(cx, W-cx)
    min_margin_y = min(cy, H-cy)
    assert min_margin_x > W/5, f'Bad principal point in view={info}'
    assert min_margin_y > H/5, f'Bad principal point in view={info}'
    # the new window will be a rectangle of size (2*min_margin_x, 2*min_margin_y) centered on (cx,cy)
    l, t = cx - min_margin_x, cy - min_margin_y
    r, b = cx + min_margin_x, cy + min_margin_y
    crop_bbox = (l, t, r, b)
    image, depthmap, intrinsics, normal, far_mask = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox, normal=normal)

    # transpose the resolution if necessary
    W, H = image.size  # new size
    # NOTE: Here we don't care about portrait image.
    # assert resolution[0] >= resolution[1]
    # if H > 1.1*W:
    #     # image is portrait mode
    #     resolution = resolution[::-1]
    # elif 0.9 < H/W < 1.1 and resolution[0] != resolution[1]:
    #     # image is square, so we chose (portrait, landscape) randomly
    #     if rng.integers(2):
    #         resolution = resolution[::-1]

    # high-quality Lanczos down-scaling
    target_resolution = np.array(resolution)
    if self.aug_focal:
        crop_scale = self.aug_focal + (1.0 - self.aug_focal) * np.random.beta(0.5, 0.5) # beta distribution, bi-modal
        image, depthmap, intrinsics, normal, far_mask = cropping.center_crop_image_depthmap(image, depthmap, intrinsics, crop_scale, normal=normal, far_mask=far_mask)

    if self.aug_crop > 1:
        target_resolution += rng.integers(0, self.aug_crop)
    image, depthmap, intrinsics, normal, far_mask = cropping.rescale_image_depthmap(image, depthmap, intrinsics, target_resolution, normal=normal, far_mask=far_mask) # slightly scale the image a bit larger than the target resolution

    # actual cropping (if necessary) with bilinear interpolation
    intrinsics2 = cropping.camera_matrix_of_crop(intrinsics, image.size, resolution, offset_factor=0.5)
    crop_bbox = cropping.bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
    image, depthmap, intrinsics2, normal, far_mask = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox, normal=normal, far_mask=far_mask)

    other = [x for x in [normal, far_mask] if x is not None]
    return image, depthmap, intrinsics2, *other
    
    

def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    # return functools.reduce(torch.matmul, matrices)
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])

def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def xyzqxqyqxqw_to_c2w(xyzqxqyqxqw):
    xyzqxqyqxqw = np.array(xyzqxqyqxqw, dtype=np.float32)
    #NOTE: we need to convert x_y_z coordinate system to z_x_y coordinate system
    z, x, y = xyzqxqyqxqw[:3]
    qz, qx, qy, qw = xyzqxqyqxqw[3:]
    c2w = np.eye(4)
    c2w[:3, :3] = np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
    ])
    c2w[:3, 3] = np.array([x, y, z])
    return c2w

def umeyama(X, Y):
    """
    Estimates the Sim(3) transformation between `X` and `Y` point sets.

    Estimates c, R and t such as c * R @ X + t ~ Y.

    Parameters
    ----------
    X : numpy.array
        (m, n) shaped numpy array. m is the dimension of the points,
        n is the number of points in the point set.
    Y : numpy.array
        (m, n) shaped numpy array. Indexes should be consistent with `X`.
        That is, Y[:, i] must be the point corresponding to X[:, i].

    Returns
    -------
    c : float
        Scale factor.
    R : numpy.array
        (3, 3) shaped rotation matrix.
    t : numpy.array
        (3, 1) shaped translation vector.
    """
    mu_x = X.mean(axis=1).reshape(-1, 1)
    mu_y = Y.mean(axis=1).reshape(-1, 1)
    var_x = np.square(X - mu_x).sum(axis=0).mean()
    cov_xy = ((Y - mu_y) @ (X - mu_x).T) / X.shape[1]
    U, D, VH = np.linalg.svd(cov_xy)
    S = np.eye(X.shape[0])
    if np.linalg.det(U) * np.linalg.det(VH) < 0:
        S[-1, -1] = -1
    c = np.trace(np.diag(D) @ S) / var_x
    R = U @ S @ VH
    t = mu_y - c * R @ mu_x
    return c, R, t


def get_pose_rank_ids(total_ids, extrinsics, expand_ratio=None, upper_bound=None):

    num_views = len(extrinsics)
    ranking, dists = compute_ranking(extrinsics, lambda_t=1.0, normalize=True, batched=True)
    # reference_view = random.sample(range(num_views), 1)[0]
    reference_view = np.random.randint(0, num_views)
    refview_ranking = ranking[reference_view]

    start_idx = 0

    # Determine the actual expand_range
    if upper_bound is None:
        # Use ratio to determine range
        expand_range = int(total_ids * expand_ratio * 2)
    else:
        expand_range = min(upper_bound, int(total_ids * expand_ratio * 2))
    high_bound = min(num_views, start_idx + expand_range)

    # Create the valid range of indices
    valid_ranking_ids = np.arange(start_idx, high_bound)
    valid_range = refview_ranking[valid_ranking_ids]

    # Sample 'total_ids - 1' items, because we already have the start_idx
    sampled_ids = np.random.choice(
        valid_range,
        size=(total_ids - 1),
        replace=True,   # we accept the situation that some sampled ids are the same
    )

    # start_randk_element = refview_ranking[start_idx]
    # return np.insert(sampled_ids, 0, start_randk_element)

    # # for nvs, farthest views must be at the start and end of the sequence
    # sampled_rankings = [np.where(refview_ranking == id_val)[0][0] for id_val in sampled_ids]
    # farthest_rank_idx = np.argmax(sampled_rankings)  # Find the element with smallest ranking (highest rank)

    # farthest_rank_element = sampled_ids[farthest_rank_idx]
    # sampled_ids = np.delete(sampled_ids, farthest_rank_idx)
    # sampled_ids = np.append(sampled_ids, farthest_rank_element)

    start_randk_element = refview_ranking[start_idx]
    result_ids = np.insert(sampled_ids, 0, start_randk_element)
    return result_ids
    

def get_nearby_ids(ids, full_seq_num,  replace=True, expand_ratio=None, expand_range=None):
    """
    TODO: add the function to sample the ids by pose similarity ranking.

    Sample a set of IDs from a sequence close to a given start index.

    You can specify the range either as a ratio of the number of input IDs
    or as a fixed integer window.


    Args:
        ids (list): Initial list of IDs. The first element is used as the anchor.
        full_seq_num (int): Total number of items in the full sequence.
        expand_ratio (float, optional): Factor by which the number of IDs expands
            around the start index. Default is 2.0 if neither expand_ratio nor
            expand_range is provided.
        expand_range (int, optional): Fixed number of items to expand around the
            start index. If provided, expand_ratio is ignored.

    Returns:
        numpy.ndarray: Array of sampled IDs, with the first element being the
            original start index.

    Examples:
        # Using expand_ratio (default behavior)
        # If ids=[100,101,102] and full_seq_num=200, with expand_ratio=2.0,
        # expand_range = int(3 * 2.0) = 6, so IDs sampled from [94...106] (if boundaries allow).

        # Using expand_range directly
        # If ids=[100,101,102] and full_seq_num=200, with expand_range=10,
        # IDs are sampled from [90...110] (if boundaries allow).

    Raises:
        ValueError: If no IDs are provided.
    """
    if len(ids) == 0:
        raise ValueError("No IDs provided.")
    if expand_ratio == -1: 
        return ids

    if expand_range is None and expand_ratio is None:
        expand_ratio = 2.0  # Default behavior

    total_ids = len(ids)
    start_idx = ids[0]

    # Determine the actual expand_range
    if expand_range is None:
        # Use ratio to determine range
        expand_range = int(total_ids * expand_ratio)

    # Calculate valid boundaries
    low_bound = max(0, start_idx - expand_range)
    high_bound = min(full_seq_num, start_idx + expand_range)
    # print(low_bound, high_bound)
    # Create the valid range of indices
    valid_range = np.arange(low_bound, high_bound)

    # Sample 'total_ids - 1' items, because we already have the start_idx
    sampled_ids = np.random.choice(
        valid_range,
        size=(total_ids - 1),
        replace=replace,   # we accept the situation that some sampled ids are the same
    )

    # Insert the start_idx at the beginning
    result_ids = np.insert(sampled_ids, 0, start_idx)

    return result_ids

def get_nearby_pose_ranking(ids, ranking, full_seq_num, expand_ratio=None, expand_range=None):
    """
    Sample a set of IDs from a sequence based on pose similarity ranking.

    Args:
        ids (list): Initial list of IDs. The first element is used as the anchor.
        ranking (numpy.ndarray): Array of pose similarity rankings.
        full_seq_num (int): Total number of items in the full sequence.
        expand_ratio (float, optional): Factor by which the number of IDs expands
            around the start index. Default is 2.0 if neither expand_ratio nor
            expand_range is provided.
        expand_range (int, optional): Fixed number of items to expand around the
            start index. If provided, expand_ratio is ignored.

    Returns:
        numpy.ndarray: Array of sampled IDs, with the first element being the
            original start index.
    """
    if len(ids) == 0:
        raise ValueError("No IDs provided.")

    if expand_range is None and expand_ratio is None:
        expand_ratio = 2.0

    total_ids = len(ids)
    start_idx = ids[0]

    # Determine the actual expand_range
    if expand_range is None:
        # Use ratio to determine range
        expand_range = int(total_ids * expand_ratio)
    # Fetch the valid ids from the ranking
    valid_ids = ranking[:expand_range]

    # Sample 'total_ids - 1' items, because we already have the start_idx
    sampled_ids = np.random.choice(
        valid_ids,
        size=(total_ids - 1),
        replace=True,   # we accept the situation that some sampled ids are the same
    )

    result_ids = np.insert(sampled_ids, 0, start_idx)

    return result_ids
        


def load_16big_png_depth(depth_png: str) -> np.ndarray:
    """
    Loads a 16-bit PNG as a half-float depth map (H, W), returning a float32 NumPy array.

    Implementation detail:
      - PIL loads 16-bit data as 32-bit "I" mode.
      - We reinterpret the bits as float16, then cast to float32.

    Args:
        depth_png (str):
            File path to the 16-bit PNG.

    Returns:
        np.ndarray:
            A float32 depth array of shape (H, W).
    """
    with Image.open(depth_png) as depth_pil:
        depth = (
            np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
            .astype(np.float32)
            .reshape((depth_pil.size[1], depth_pil.size[0]))
        )
    return depth


def save_ply_visualization(
    pred_dict, cur_step, save_name, init_conf_threshold=20.0, save_depth=True, save_point=False, save_gt=True, save_images=True, debug=False, all_batch=False,gt_only=False,
):
    """
    Saves a PLY visualization of the world points and images.
    Args:
    """
    pred_dict = pred_dict.copy()
    for key in pred_dict.keys():
        if isinstance(pred_dict[key], torch.Tensor):
            pred_dict[key] = pred_dict[key].detach()
    index = np.random.randint(0, len(pred_dict["images"]))
    seq_len = len(pred_dict["images"][index])  

    seq_lens = []
    indexs = []
    if debug:
        for _ in range(3):
            index = np.random.randint(0, len(pred_dict["images"]))
            seq_len = len(pred_dict["images"][index])  
            indexs.append(index)
            seq_lens.append(seq_len)
    elif all_batch:
        for id in range(len(pred_dict["images"])):
            indexs.append(id)
            seq_lens.append(len(pred_dict["images"][id]))
    else:
        seq_lens = [seq_len]
        indexs = [index]
    
    for index, seq_len in zip(indexs, seq_lens):

        if save_depth:
    
            gt_pts = pred_dict['world_points'][index].float().cpu().numpy()   

            valid_mask= pred_dict['point_masks'][index].cpu().numpy()  

            images = pred_dict["images"][index] 

            view_info =  pred_dict["view_infos"][index].split('/')[0]
            
            if not gt_only:
                pred_pts = pred_dict["points"][index]  
    
                data_h, data_w         = images.shape[-2:]
                global_points = pred_pts

                data_size = (data_h, data_w)
                global_points = F.interpolate(
                    global_points.permute(0, 3, 1, 2), data_size,
                    mode="bilinear", align_corners=False, antialias=True
                ).permute(0, 2, 3, 1)  # align to gt
                
                global_points =  global_points.cpu().numpy()
                pred_pts = global_points

                assert pred_pts.shape == gt_pts.shape, f"Predicted points shape {pred_pts.shape} does not match ground truth shape {gt_pts.shape}."

            colors = images.permute(0, 2, 3, 1)[valid_mask].cpu().numpy().reshape(-1, 3)
            if not gt_only:
                # 5. coarse align
                c, R, t = umeyama(pred_pts[valid_mask].T, gt_pts[valid_mask].T)
                pred_pts = c * np.einsum('nhwj, ij -> nhwi', pred_pts, R) + t.T

                # 6. filter invalid points
                pred_pts = pred_pts[valid_mask].reshape(-1, 3)

            gt_pts = gt_pts[valid_mask].reshape(-1, 3)

            # 7. save predicted & ground truth point clouds
            os.makedirs(osp.join('save_path', save_name), exist_ok=True)
            if not gt_only:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pred_pts)
                pcd.colors = o3d.utility.Vector3dVector(colors)
                o3d.io.write_point_cloud(osp.join('save_path', save_name, f"{view_info}_{cur_step}_predicted_pointcloud_numI{seq_len}_{index}.ply"), pcd)

            pcd_gt = o3d.geometry.PointCloud()
            pcd_gt.points = o3d.utility.Vector3dVector(gt_pts)
            pcd_gt.colors = o3d.utility.Vector3dVector(colors)
            o3d.io.write_point_cloud(osp.join('save_path', save_name, f"{view_info}_{cur_step}_GT_pointcloud_numI{seq_len}_{index}.ply"), pcd_gt)

        if save_images:
            # Save images as a grid
            images = pred_dict["images"][index]
            images_grid = torchvision.utils.make_grid(
                images, nrow=8, normalize=False, scale_each=False
            )
            images_grid = images_grid
            images_save_path = os.path.join('save_path', save_name, f"{view_info}_{cur_step}_gt_images_{seq_len}_{index}.png")
            os.makedirs(os.path.dirname(images_save_path), exist_ok=True)
            torchvision.utils.save_image(
                images_grid, images_save_path, normalize=False, scale_each=False
            )
            

def save_ply_visualization_cpu_batch(
    pred_dict, cur_step, save_name, init_conf_threshold=20.0, save_depth=True, save_point=False, save_gt=True, save_images=True, debug=False, all_batch=False,gt_only=False,
):
    """
    Saves a PLY visualization of the world points and images.
    Args:
    """
  
    seq_len = len(pred_dict["images"])  

    if save_depth:
    
        gt_pts = np.array(pred_dict['world_points'])#
        
        valid_mask= pred_dict['point_masks']#
        valid_mask = np.array(valid_mask)
        images = pred_dict["images"]  #
        images = torch.from_numpy(np.stack(images).astype(np.float32)).contiguous().permute(0, 3, 1, 2).to(torch.float32).div(255)
        view_info =  pred_dict["view_infos"][0].split('/')[0]

        colors = images.permute(0, 2, 3, 1)[valid_mask].numpy().reshape(-1, 3)
        gt_pts = gt_pts[valid_mask].reshape(-1, 3)

        # 7. save predicted & ground truth point clouds
        os.makedirs(osp.join('save_path', save_name), exist_ok=True)
        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(gt_pts)
        pcd_gt.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(osp.join('save_pathg', save_name, f"{view_info}_{cur_step}_GT_pointcloud_numI{seq_len}.ply"), pcd_gt)

    if save_images:
        # Save images as a grid
        images = np.array(pred_dict["images"])
        images = torch.from_numpy(np.stack(images).astype(np.float32)).contiguous().permute(0, 3, 1, 2).to(torch.float32).div(255)

        images_grid = torchvision.utils.make_grid(
            images, nrow=8, normalize=False, scale_each=False
        )
        images_grid = images_grid
        images_save_path = os.path.join('save_path', save_name, f"{view_info}_{cur_step}_gt_images_{seq_len}.png")
        os.makedirs(os.path.dirname(images_save_path), exist_ok=True)
        torchvision.utils.save_image(
            images_grid, images_save_path, normalize=False, scale_each=False
        )

def save_ply(points, colors, filename):
    import open3d as o3d                
    if torch.is_tensor(points):
        points_visual = points.reshape(-1, 3).cpu().numpy()
    else:
        points_visual = points.reshape(-1, 3)
    if torch.is_tensor(colors):
        points_visual_rgb = colors.reshape(-1, 3).cpu().numpy()
    else:
        points_visual_rgb = colors.reshape(-1, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_visual.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(points_visual_rgb.astype(np.float64))
    pcd = pcd.uniform_down_sample(every_k_points=4)
    o3d.io.write_point_cloud(filename, pcd, write_ascii=True)

def value_to_decimal(value, decimal_places):
    decimal.getcontext().rounding = decimal.ROUND_HALF_UP  # define rounding method
    return decimal.Decimal(str(float(value))).quantize(
        decimal.Decimal("1e-{}".format(decimal_places))
    )


def read_traj_local(traj_path):
    quaternions = []
    poses = []
    timestamps = []
    poses_p_to_w = []
    with open(traj_path) as f:
        traj_lines = f.readlines()
        for line in traj_lines:
            tokens = line.split()
            assert len(tokens) == 7
            traj_timestamp = float(tokens[0])

            timestamps_decimal_value = value_to_decimal(traj_timestamp, 3)
            timestamps.append(
                float(timestamps_decimal_value)
            )  # for spline interpolation

            angle_axis = [float(tokens[1]), float(tokens[2]), float(tokens[3])]
            r_w_to_p, _ = cv2.Rodrigues(np.asarray(angle_axis))
            t_w_to_p = np.asarray(
                [float(tokens[4]), float(tokens[5]), float(tokens[6])]
            )

            pose_w_to_p = np.eye(4)
            pose_w_to_p[:3, :3] = r_w_to_p
            pose_w_to_p[:3, 3] = t_w_to_p

            pose_p_to_w = np.linalg.inv(pose_w_to_p)

            # r_p_to_w_as_quat = quaternion.from_rotation_matrix(pose_p_to_w[:3, :3])
            t_p_to_w = pose_p_to_w[:3, 3]
            poses_p_to_w.append(pose_p_to_w)
            poses.append(t_p_to_w)
            # quaternions.append(r_p_to_w_as_quat)
    return timestamps, poses, None, poses_p_to_w

