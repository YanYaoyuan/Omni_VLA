# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import traceback
from PIL import Image, ImageFile, PngImagePlugin
import numpy as np 

from .data_utils import pil_img2rgb
from .distributed_iterable_dataset import DistributedIterableDataset
import random 
import cv2
import logging
from .dataset_utils_vggt import *
import torch.distributed as dist
import torch
import pi3.utils.cropping as cropping
from pi3.utils.geometry import depthmap_to_absolute_camera_coordinates
from .frame_sampling_utils import compute_ranking
import gzip 
from pathlib import Path
from scipy.spatial.transform import Rotation

Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte
IMAGE_PER_SEQ = 2


CO3D_SEEN_CATEGORIES = [
    "apple",
    "backpack",
    "banana",
    "baseballbat",
    "baseballglove",
    "bench",
    "bicycle",
    "bottle",
    "bowl",
    "broccoli",
    "cake",
    "car",
    "carrot",
    "cellphone",
    "chair",
    "cup",
    "donut",
    "hairdryer",
    "handbag",
    "hydrant",
    "keyboard",
    "laptop",
    "microwave",
    "motorcycle",
    "mouse",
    "orange",
    "parkingmeter",
    "pizza",
    "plant",
    "stopsign",
    "teddybear",
    "toaster",
    "toilet",
    "toybus",
    "toyplane",
    "toytrain",
    "toytruck",
    "tv",
    "umbrella",
    "vase",
    "wineglass",
]


def check_valid_numpy(array, name):
    """
    Checks if a numpy array is valid (not None and has the expected shape).
    Raises an error if the array is None or has an unexpected shape.
    """
    if array is None:
        print(f"{name} is None")
        return True
    
    if not isinstance(array, np.ndarray):
        print(f"{name} must be a numpy.ndarray, got {type(array)}")
        return True
    
    if array.size == 0:
        print(f"{name} is empty")
        return True

    if np.isnan(array).any():
        print(f"{name} contains NaN values")
        return True

    if np.isinf(array).any():
        print(f"{name} contains Inf values")
        return True

    return False  # No exception raised, array is valid


class SftJSONLIterableReconDataset(DistributedIterableDataset):
    def __init__(
        self, dataset_name, dino_transform, tokenizer, frame_sampler, 
        jsonl_path_list, data_dir_list, num_used_data, 
        local_rank=0, world_size=1, num_workers=8, data_status=None, 
        shuffle_lines=False, shuffle_seed=0,
    ):
        """
        jsonl_path_list: list of jsonl file paths
        data_dir_list: list of image directories containing the images of each jsonl file
        num_used_data: list of number of sampled data points for each jsonl
        """
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        # self.transform = transform
        # self.vit_transform = vit_transform
        self.dino_transform = dino_transform
        self.tokenizer = tokenizer
        self.frame_sampler = frame_sampler
        self.data_status = data_status

        self.img_size = 518 ### second stage 
        self.use_dinov3 = False   
        if self.use_dinov3:
            self.patch_size = 16
        else:
            self.patch_size = 14
        if self.use_dinov3:
            self.img_size = 512 # 

        self.aug_scale = [0.8, 1.2] 
        self.rescale = True
        self.rescale_aug = True
        self.landscape_check = False  #True
        self.training = True # hardcode 
        self.enable_random_image_num = True
        self.ceph_read = True

        self._rng = np.random.default_rng(shuffle_seed)

        self.aug_crop = 16 ###aug_crop
        self.aug_focal = 0.9 ####aug_focal
        self.z_far = 0 ####z_far
        self.random_sample_thres = 0.1 #random_sample_thre

        self.data_paths = self.get_data_paths(
            jsonl_path_list, 
            data_dir_list, 
            num_used_data, 
            shuffle_lines, 
            shuffle_seed,
        )
        self.base_seed = shuffle_seed
        self.random_image_num = 0
        self.frame_num = 0
        self.random_aspect_ratio = 1.0
        self.resolution = [224, 224]
        if self.use_dinov3:
            self.resolution = [256, 256]

        self.set_epoch()

        self.scannet_invalid_list = 'scannet_recon_invalid_list.json'
        with open(self.scannet_invalid_list, 'r') as f:
            self.scannet_invalid_list = json.load(f)

    def set_random_image_num(self, num):
        self.random_image_num = num     
        self.frame_num = num

    def set_random_aspect_ratio(self, num):
        self.random_aspect_ratio = num     
  
    def set_step_rng(self, rng):
        self._rng = np.random.default_rng(rng)     

    def convert_intrinsics(self, meta_data):
        store_h, store_w = meta_data["h"], meta_data["w"]
        fx, fy, cx, cy = (
            meta_data["fl_x"],
            meta_data["fl_y"],
            meta_data["cx"],
            meta_data["cy"],
        )
        intrinsics = np.eye(3, dtype=np.float32)
        intrinsics[0, 0] = float(fx) / 4.0 # downsample by 4
        intrinsics[1, 1] = float(fy) / 4.0
        intrinsics[0, 2] = float(cx) / 4.0
        intrinsics[1, 2] = float(cy) / 4.0
        return intrinsics

    def blender2opencv_c2w(self, pose):
        blender2opencv = np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        )
        opencv_c2w = np.array(pose) @ blender2opencv
        return opencv_c2w.tolist()     

    def get_data_paths(
        self, 
        jsonl_path_list, 
        data_dir_list, 
        num_used_data, 
        shuffle_lines, 
        shuffle_seed,
    ):
        data_paths = []
        for jsonl_path, image_dir, num_data_point in zip(
            jsonl_path_list, data_dir_list, num_used_data
        ):
            with open(jsonl_path, 'r') as f:
                raw_data = f.readlines()
            if shuffle_lines: 
                self.rng.seed(shuffle_seed)
                self.rng.shuffle(raw_data)
            raw_data = raw_data[:num_data_point]
            data_paths.extend([(json_data, image_dir) for json_data in raw_data])
        return data_paths

    def change_format(self, data, num_images):
        elements = []
        for conversation in data['conversations']:
            if conversation['from'] == 'human':
                if '<image>' not in conversation['value']:
                    elements.append({
                        'type': 'text',
                        'has_loss': 0,
                        'text': conversation['value'],
                    })
                else:

                    text_list = conversation['value'].split('<image>')
                    for idx, text in enumerate(text_list):
                        if text.strip() != '':
                            elements.append({
                                'type': 'text',
                                'has_loss': 0,
                                'text': text.strip(),
                            })
                        if (idx != len(text_list) - 1) and (idx < num_images):
                            elements.append({'type': 'image',})
            elif conversation['from'] == 'gpt':
                elements.append({
                    'type': 'text',
                    'has_loss': 1,
                    'text': conversation['value'],
                })
        return elements

    def _crop_resize_if_necessary(self, image, depthmap, intrinsics, resolution, rng=None, info=None, normal=None, far_mask=None):
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
     

    def get_nearby_ids(self, ids, full_seq_num, expand_ratio=None, expand_range=None, rng=None):
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

        # Create the valid range of indices
        valid_range = np.arange(low_bound, high_bound)

        # Sample 'total_ids - 1' items, because we already have the start_idx
        sampled_ids = np.random.choice(
            valid_range,
            size=(total_ids - 1),
            replace=True,   # we accept the situation that some sampled ids are the same
        )

        # Insert the start_idx at the beginning
        result_ids = np.insert(sampled_ids, 0, start_idx)

        return result_ids
        
    def get_pose_rank_ids(self, total_ids, extrinsics, expand_ratio=None, upper_bound=None):
    
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
      


    def get_nearby_pose_ranking(self, ids, ranking, full_seq_num, expand_ratio=None, expand_range=None):
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
            

    def get_target_shape(self, aspect_ratio):
        """
        Calculate the target shape based on the given aspect ratio.
        
        Args:
            aspect_ratio: Target aspect ratio
            
        Returns:
            numpy.ndarray: Target image shape [height, width]
        """
        short_size = int(self.img_size * aspect_ratio)
        small_size = self.patch_size

        # ensure the input shape is friendly to vision transformer
        if short_size % small_size != 0:
            short_size = (short_size // small_size) * small_size

        image_shape = np.array([short_size, self.img_size])
        return image_shape


    def process_one_image(
        self,
        image,
        depth_map,
        extri_opencv,
        intri_opencv,
        original_size,
        target_image_shape,
        track=None,
        filepath=None,
        safe_bound=4,
    ):
        """
        Process a single image and its associated data.

        This method handles image transformations, depth processing, and coordinate conversions.

        Args:
            image (numpy.ndarray): Input image array
            depth_map (numpy.ndarray): Depth map array
            extri_opencv (numpy.ndarray): Extrinsic camera matrix (OpenCV convention)
            intri_opencv (numpy.ndarray): Intrinsic camera matrix (OpenCV convention)
            original_size (numpy.ndarray): Original image size [height, width]
            target_image_shape (numpy.ndarray): Target image shape after processing
            track (numpy.ndarray, optional): Optional tracking information. Defaults to None.
            filepath (str, optional): Optional file path for debugging. Defaults to None.
            safe_bound (int, optional): Safety margin for cropping operations. Defaults to 4.

        Returns:
            tuple: (
                image (numpy.ndarray): Processed image,
                depth_map (numpy.ndarray): Processed depth map,
                extri_opencv (numpy.ndarray): Updated extrinsic matrix,
                intri_opencv (numpy.ndarray): Updated intrinsic matrix,
                world_coords_points (numpy.ndarray): 3D points in world coordinates,
                cam_coords_points (numpy.ndarray): 3D points in camera coordinates,
                point_mask (numpy.ndarray): Boolean mask of valid points,
                track (numpy.ndarray, optional): Updated tracking information
            )
        """
        # Make copies to avoid in-place operations affecting original data
        image = np.copy(image)
        depth_map = np.copy(depth_map)
        extri_opencv = np.copy(extri_opencv)
        intri_opencv = np.copy(intri_opencv)
        if track is not None:
            track = np.copy(track)

        # Apply random scale augmentation during training if enabled
        if self.training and self.aug_scale:
            random_h_scale, random_w_scale = np.random.uniform(
                self.aug_scale[0], self.aug_scale[1], 2
            )
            # Avoid random padding by capping at 1.0
            random_h_scale = min(random_h_scale, 1.0)
            random_w_scale = min(random_w_scale, 1.0)
            aug_size = original_size * np.array([random_h_scale, random_w_scale])
            aug_size = aug_size.astype(np.int32)
        else:
            aug_size = original_size

        # Move principal point to the image center and crop if necessary
        image, depth_map, intri_opencv, track = crop_image_depth_and_intrinsic_by_pp(
            image, depth_map, intri_opencv, aug_size, track=track, filepath=filepath,
        )

        original_size = np.array(image.shape[:2])  # update original_size
        target_shape = target_image_shape

        # Handle landscape vs. portrait orientation
        rotate_to_portrait = False
        if self.landscape_check:
            # Switch between landscape and portrait if necessary
            if original_size[0] > 1.25 * original_size[1]:
                if (target_image_shape[0] != target_image_shape[1]) and (np.random.rand() > 0.5):
                    target_shape = np.array([target_image_shape[1], target_image_shape[0]])
                    rotate_to_portrait = True

        # Resize images and update intrinsics
        if self.rescale:
            image, depth_map, intri_opencv, track = resize_image_depth_and_intrinsic(
                image, depth_map, intri_opencv, target_shape, original_size, track=track,
                safe_bound=safe_bound,
                rescale_aug=self.rescale_aug
            )
        else:
            print("Not rescaling the images")

        # Ensure final crop to target shape
        image, depth_map, intri_opencv, track = crop_image_depth_and_intrinsic_by_pp(
            image, depth_map, intri_opencv, target_shape, track=track, filepath=filepath, strict=True,
        )

        # Apply 90-degree rotation if needed
        if rotate_to_portrait:
            assert self.landscape_check
            clockwise = np.random.rand() > 0.5
            image, depth_map, extri_opencv, intri_opencv, track = rotate_90_degrees(
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                clockwise=clockwise,
                track=track,
            )

        # Convert depth to world and camera coordinates
        world_coords_points, cam_coords_points, point_mask = (
            depth_to_world_coords_points(depth_map, extri_opencv, intri_opencv)
        )

        return (
            image,
            depth_map,
            extri_opencv,
            intri_opencv,
            world_coords_points,
            cam_coords_points,
            point_mask,
            track,
        )

    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if self.data_status is not None:
            row_start_id = self.data_status[worker_id] + 1
        else:
            row_start_id = 0
        
        #comment: btw this is for all datasets in same group. 

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at row#{row_start_id}"
        )

        while True:
            data_paths_per_worker_ = data_paths_per_worker[row_start_id:]

            allow_retry_times = 50 #20 
            retry_time = 0
            data_fail = False 
            error = None
            pi3 = True
            if pi3:
                shuffle_seq_views = True

            for row_idx, (data, image_dir) in enumerate(data_paths_per_worker_, start=row_start_id):
                
                num_tokens = 0
                dino_image_tensor_list = []
                dino_thw = []
                dino_images =[]
                text_ids_list = []
                sequence_plan = []

                images = []
                depths = []
                cam_points = []
                world_points = []
                point_masks = []
                extrinsics = []
                intrinsics = []
                # image_paths = []
                view_infos = []
                original_sizes = []
                img_per_seq_list = []
                img_per_seq =  self.frame_num
                # print('img_per_seq', img_per_seq)

                if high_resolution_training:
                    aspect_ratio = self.random_aspect_ratio 
                else:
                    aspect_ratio = 1.0
                target_image_shape = self.get_target_shape(aspect_ratio)
                self.resolution = target_image_shape

                try:
                    data_item = json.loads(data)
                    if 'meta' in data_item:
                        data_meta = data_item['meta']
                
                    data_scene_name = data_item['scene_name']
                    this_scene = data_item['seq_name']
                    text_ins = 'Reconstruct the 3D scene.'
                    rng = self._rng

                    if data_scene_name == 'scannet':
                        num_imgs = data_item['num_images']
                        image_dir = data_item['img_dir']

                        valid_idxs = [i for i in range(num_imgs) if i not in self.scannet_invalid_list[this_scene]]
                        num_imgs = len(valid_idxs)

                        if self.frame_num > 16 and rng.random() < self.random_sample_thres:
                            all_keys = valid_idxs
                            should_replace = len(all_keys) < self.frame_num
                            idxs = list(rng.choice(all_keys, size=self.frame_num, replace=should_replace))
                        else:
                            idxs = [rng.integers(0, num_imgs)]
                            
                            scannet_max_distance=240
                            max_distance = int(scannet_max_distance / 8 * self.frame_num)
                            start_idx = max(0, idxs[-1] - max_distance)
                            end_idx = min(num_imgs-1, start_idx + 2*max_distance)
                            start_idx = max(0, end_idx - 2*max_distance)
                            valid_indices = np.arange(start_idx, end_idx + 1)

                            if rng.random() < 0.5:
                                should_replace = len(valid_indices) < self.frame_num - 1
                                idxs.extend(list(rng.choice(valid_indices, self.frame_num-1, replace=should_replace)))
                                idxs = [valid_idxs[i] for i in idxs]
                            else:
                                ref_frame_val = idxs[0]
                                num_additional_to_select = self.frame_num - 1
                                additional_selected_values = []
                                pool_for_others_values = [val for val in valid_indices]
                                pool_for_others_values.sort() 

                                should_replace_for_others = len(pool_for_others_values) < num_additional_to_select

                                if not pool_for_others_values: 
                                    if should_replace_for_others:
                                        additional_selected_values = [ref_frame_val] * num_additional_to_select
                                else:
                                    if not should_replace_for_others and len(pool_for_others_values) >= num_additional_to_select:
                                        strata = np.array_split(pool_for_others_values, num_additional_to_select+1)
                                        for stratum in strata:
                                            if len(stratum) > 0 and ref_frame_val not in stratum: 
                                                additional_selected_values.append(rng.choice(stratum))
                                    else:
                                        additional_selected_values = list(rng.choice(
                                            pool_for_others_values,
                                            num_additional_to_select,
                                            replace=(should_replace_for_others or (len(pool_for_others_values) < num_additional_to_select))
                                        ))

                                idxs = [ref_frame_val, *additional_selected_values]
                                idxs = [valid_idxs[idx] for idx in idxs]

                        views = []
                        for idx in idxs:
                            path_idx = str(idx).zfill(5)
                            image_path =os.path.join( image_dir, path_idx + '.jpg')
                            depth_path = os.path.join(image_dir, path_idx + '.png')

                            rgb_image = np.array(Image.open(image_path).resize((640, 480), resample=lanczos))
                            
                            with Image.open(depth_path) as depth_img:
                                depth_map = np.array(depth_img).astype(np.int32) / 1000.0
                            
                            pose_path = os.path.join(image_dir, path_idx + '.txt')
                            extri_opencv = np.loadtxt(pose_path).astype(np.float32).reshape(4, 4)  
                            intri_depth = os.path.join(image_dir, "depth_intrinsic.txt")
                            intri_opencv = np.loadtxt(intri_depth).astype(np.float32).reshape(4, 4)[:3, :3] #3 x 3
                            rgb_image, depth_map, intrinsic_ = self._crop_resize_if_necessary(
                                rgb_image, depth_map, intri_opencv.copy(), self.resolution, rng=rng, info=image_path)
                            
                            images.append(rgb_image)
                            depths.append(depth_map)
                            extrinsics.append(extri_opencv.astype(np.float32))
                            intrinsics.append(intrinsic_.astype(np.float32))
                    
                            view_infos.append(f'{data_scene_name}/{this_scene}/{str(idx)}')
        

                    #######################begin data type checking ################################### pi3

                    if shuffle_seq_views:
                        indices = list(range(len(images)))
                        self._rng.shuffle(indices)
                        images = [images[i] for i in indices]
                        depths = [depths[i] for i in indices]
                        extrinsics = [extrinsics[i] for i in indices]
                        intrinsics = [intrinsics[i] for i in indices]
                        view_infos = [view_infos[i] for i in indices]

                    new_depths = []
                    world_points = []
                    point_masks = []
                    skip_this_scene = False 
                    for v,(img,depthmap,camera_pose, camera_intrinsics, view_info) in enumerate(zip(images,depths,extrinsics,intrinsics, view_infos)):

                        width, height = img.size

                        assert np.isfinite(camera_pose).all(), f'NaN in camera pose for view {view_info}'
                    
                        assert np.isfinite(depthmap).all(), f'NaN in depthmap for view {view_info}'

                        scene_label = view_info.split('/')[0]
                
                        if scene_label in ['co3dv2', 'wildrgbd', 'blendedmvs']: 
                            z_far = 0
                        elif scene_label in ['gtasfm', 'matrixcity', "taskonomy", 'hypersim', 'nav_20w', 'vkitti', 'megadepth', 'dl3dv', 'omniworld', 'unreal4k']: ##already process in its dataset
                            z_far = 0
                        elif scene_label in ['tartanair', 'scannet']:
                            z_far = 80 
                        elif scene_label in ['scannetpp', 'arkitscenes']:
                            z_far = 120
                        else:
                            z_far = 0

                        pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(depthmap, camera_intrinsics, camera_pose, z_far=z_far)

                        valid_mask = valid_mask & np.isfinite(pts3d).all(axis=-1)
                        depthmap[~valid_mask] = 0.0

                        if not valid_mask.sum() > 0:
                            skip_this_scene = True 
                            break 

                        
                        assert valid_mask.sum() > 0, f"viewinfo{view_info}, depthmap{depthmap}"

                        new_depths.append(depthmap)
                        world_points.append(pts3d)
                        point_masks.append(valid_mask)

                    if skip_this_scene:
                        print(f'skipping scene: {view_infos}')
                        continue

                except Exception as e:
                    data_fail = True
                    retry_time += 1
                    error = e
                    print(
                        f"Failed to load data from ({view_infos}) for error {e}.", flush=True
                    )
                    traceback.print_exc()
                    continue
                
                raw_images = images 
                transform_stride =self.patch_size 
                for raw_image in raw_images:
                    image_tensor = self.dino_transform(raw_image, img_num=len(raw_images)) 
                    dino_images.append(raw_image)
                    dino_image_tensor_list.append(image_tensor)
                    height, width = image_tensor.shape[1:]
                    num_tokens += width * height // transform_stride ** 2

                    grid_t = 1  
                    grid_h, grid_w = height // self.patch_size, width // self.patch_size
                    thw = torch.tensor([grid_t, grid_h, grid_w], dtype=torch.long) 
                    dino_thw.append(thw)

                text_data = text_ins
                text_ids = self.tokenizer.encode(text_data)
                if len(text_ids) > 0:
                    text_ids_list.append(text_ids)
                    num_tokens += len(text_ids)
                    current_plan = {
                        'type': 'text',
                        'enable_cfg': 0,
                        'loss': 0,
                        'special_token_loss': 0,
                        'special_token_label': None,
                    }
                    sequence_plan.append(current_plan)
                for _ in range(len(dino_image_tensor_list)):
                    current_plan = {
                        'type': 'dino_image',
                        'enable_cfg': 0,
                        'loss': 0,
                        'special_token_loss': 0,
                        'special_token_label': None,
                    }
                    sequence_plan.append(current_plan)

                if retry_time >= allow_retry_times:
                    raise error

                yield dict(
                    dino_image_tensor_list=dino_image_tensor_list,
                    dino_thw=dino_thw,
                    dino_images =dino_images,
                    text_ids_list=text_ids_list,
                    sequence_plan=sequence_plan,
                    num_tokens=num_tokens,
                    depths=new_depths,  
                    extrinsics=extrinsics,
                    intrinsics=intrinsics,
                    cam_points=cam_points,
                    world_points=world_points,
                    point_masks=point_masks,
                    view_infos=view_infos,
                    img_per_seq = img_per_seq, 
                    data_indexes={
                        "data_indexes": row_idx,
                        "worker_id": worker_id,
                        "dataset_name": self.dataset_name,
                    }
                )

            row_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")
