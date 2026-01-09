import io
import random
import os 
from PIL import Image, ImageFile, PngImagePlugin
from idna import intranges_contain

from .interleave_t2i_dataset import InterleavedBaseIterableDataset, ParquetStandardIterableDataset
from ..data_utils import pil_img2rgb, apply_template_qwenvl2, apply_template_qwenvl2_reconThenUnd
from ..distributed_iterable_dataset import DistributedIterableDataset
from ..dataset_utils_vggt import *
import torch
from copy import deepcopy
import traceback
import pi3.utils.cropping as cropping
from pi3.utils.geometry import depthmap_to_absolute_camera_coordinates
from .draw_marker import DRAW_FUNCTIONS
import torch.distributed as dist
import ast
Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

class ReconthenUndIterableDataset(ParquetStandardIterableDataset, DistributedIterableDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.img_size = 224  # 518
        self.patch_size = 14
        self.aug_scale = [0.8, 1.2] 
        self.rescale = True
        self.rescale_aug = True
        self.landscape_check = False  # True
        self.training = True  # Consider making this configurable
        self._rng = np.random.default_rng(self.shuffle_seed)
        self.random_sample_thres = 0.1 
        self.aug_crop = 0 
        self.aug_focal = None 
        self.z_far = 0  
        self.frame_num = 8 ###harcode 
        self.random_aspect_ratio = 1.0
        self.resolution = [518, 518]  
        self.spar5m_get_nearby_ids = False

    def pop_first(self, arr):
        first_val = arr[0]
        new_arr = arr[1:]
        return first_val, new_arr
    def set_random_image_num(self, num):
        self.random_image_num = num     
        self.frame_num = num
    def set_random_aspect_ratio(self, num):
        self.random_aspect_ratio = num     
    def set_step_rng(self, rng):
        self._rng = np.random.default_rng(rng)

    def draw_image(self, image, data_item):
        draw_fn = DRAW_FUNCTIONS.get(data_item.get('type', None))
        if draw_fn is None:
            print(data_item)
            raise ValueError(f"Unsupported data type: {data_item.get('type', None)}")
        draw_fn(image, data_item)


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


    def _init_data(self):
        data = {
            'sequence_plan': [],
            'text_ids_list': [],
            'image_tensor_list': [],
            'image_grid_thw_list': [],
            'dino_images': [],
            'depths': [],
            'extrinsics': [],
            'intrinsics': [],
            'cam_points': [],
            'world_points': [],
            'point_masks': [],
            'original_sizes': [],
            'dino_image_tensor_list': [],
            'dino_thw': [],
            'img_per_seq': 0,
            'num_tokens': 0,
            'new_depths': [],
            'view_infos': [],
            'image_paths': [],
        }
        return data

    def _add_text(self, data, text, need_loss, enable_cfg=True):
        text_ids = self.tokenizer.encode(text)
        data['num_tokens'] += len(text_ids)
        data['text_ids_list'].append(text_ids)
        data['sequence_plan'].append(
            {
                'type': 'text',
                'enable_cfg': 0,  #int(enable_cfg),
                'loss': int(need_loss),
                'special_token_loss': 0,
                'special_token_label': None,
            }
        )
        return data

    def _add_image(self, data, image, dino_meta, need_loss, need_dino, need_vit, enable_cfg=True, rng=None, view_info=None ):
        assert need_loss or need_dino or need_vit

        if need_dino:
            aspect_ratio = 1.0 
            depth_path = dino_meta['depth']
            if dino_meta['scene_name'] == 'matterport3d':
                with Image.open(depth_path) as depth_img:
                    depth_map = np.array(depth_img).astype(np.int32) / 4000.0
                depth_map[~np.isfinite(depth_map)] = 0

                threshold = (
                    np.percentile(depth_map[depth_map > 0], 98)
                    if depth_map[depth_map > 0].size > 0
                    else 0
                    )
                depth_map[depth_map > threshold] = 0.0

                extri_opencv = dino_meta['pose']
                extri_opencv = np.array(extri_opencv).reshape((4,4))
                intri_opencv = np.array(dino_meta['intri'])[:3, :3] #3 x 3
                
            elif dino_meta['scene_name'] == 'scannet': 
                with Image.open(depth_path) as depth_img:
                    depth_map = np.array(depth_img).astype(np.int32) / 1000.0
                
                depth_map[~np.isfinite(depth_map)] = 0
                if depth_map.shape[0] != image.shape[0]:
                    image = cv2.resize(image, (depth_map.shape[1], depth_map.shape[0]), interpolation=cv2.INTER_LINEAR)
                extri_opencv = dino_meta['pose'] # 3 x 4 
                extri_opencv = np.array(extri_opencv).reshape((4,4))
                intri_opencv = np.array(dino_meta['intri'])[:3, :3] #3 x 3
            elif dino_meta['scene_name'] == '3rscan': 
                with Image.open(depth_path) as depth_img:
                    depth_map = np.array(depth_img).astype(np.int32) / 1000.0
                depth_map[~np.isfinite(depth_map)] = 0
            
                extri_opencv = dino_meta['pose'] # 3 x 4 
                extri_opencv = np.array(extri_opencv).reshape((4,4))
                intri_opencv = np.array(dino_meta['intri'])[:3, :3] #3 x 3
            elif dino_meta['scene_name'] == 'scannetpp': 
                with Image.open(depth_path) as depth_img:
                    depth_map = np.array(depth_img).astype(np.int32) / 1000.0
                depth_map[~np.isfinite(depth_map)] = 0
                if depth_map.shape[0] != image.shape[0] or depth_map.shape[1] != image.shape[1]:
                    depth_map = cv2.resize(
                        depth_map, 
                        (image.shape[1], image.shape[0]),  # (width, height) for OpenCV
                        interpolation=cv2.INTER_NEAREST  # Better for depth maps to preserve sharpness
                    )
       
                extri_opencv = dino_meta['pose'] # 3 x 4 
                extri_opencv = np.array(extri_opencv).reshape((4,4))
                intri_opencv = np.array(dino_meta['intri'])[:3, :3] #3 x 3
            elif dino_meta['scene_name'] == 'structured3d': 
                with Image.open(depth_path) as depth_img:
                    depth_map = np.array(depth_img).astype(np.int32) / 1000.0
                depth_map[~np.isfinite(depth_map)] = 0
        
                extri_opencv = dino_meta['pose'] # 3 x 4 
                extri_opencv = np.array(extri_opencv).reshape((4,4))
                extri_opencv[:3, 3] =  extri_opencv[:3, 3] / 1000.0
                intri_opencv = np.array(dino_meta['intri'])[:3, :3] #3 x 3

            original_size = np.array(image.shape[:2])

            image, depth_map, intrinsic_ = self._crop_resize_if_necessary(
                image, depth_map, intri_opencv.copy(), self.resolution, rng=rng, info=view_info)
                                
            data['dino_images'].append(image)
            data['depths'].append(depth_map.astype(np.float32))
            data['extrinsics'].append(extri_opencv.astype(np.float32))
            data['intrinsics'].append(intrinsic_.astype(np.float32))
            data['original_sizes'].append(original_size)
            data['view_infos'].append(view_info)


            width, height = image.size
            if dino_meta['scene_name'] in ['scannet']:
                z_far = 80   
            elif dino_meta['scene_name'] in ['matterport3d']:
                z_far = 80   
            elif dino_meta['scene_name'] in ['3rscan']:
                z_far = 80    
            else:
                z_far = 80    
        
            assert np.isfinite(extri_opencv).all(), f'NaN in camera pose for view {view_info}'
            assert np.isfinite(depth_map).all(), f'NaN in depthmap for view {view_info}'

            pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(depth_map, intrinsic_, extri_opencv, z_far=z_far)
            valid_mask = valid_mask & np.isfinite(pts3d).all(axis=-1)
            depth_map[~valid_mask] = 0.0
            assert valid_mask.sum() > 0, f"viewinfo{view_info}, depthmap{depth_map}"

            data['new_depths'].append(depth_map)
            data['world_points'] .append(pts3d)
            data['point_masks'] .append(valid_mask)
            
            transform_stride =14 #hardcode, 
  
            image_tensor = self.dino_transform(image, img_num=1) 
            data['dino_image_tensor_list'].append(image_tensor)
            height, width = image_tensor.shape[1:]
            data['num_tokens'] += width * height // transform_stride ** 2

            grid_t = 1  #patches.shape[0] // self.temporal_patch_size
            grid_h, grid_w = height // self.patch_size, width // self.patch_size
            thw_dino = torch.tensor([grid_t, grid_h, grid_w], dtype=torch.long) # Shape: (3,)
            data['dino_thw'].append(thw_dino)

            data['sequence_plan'].append(
                {
                    'type': 'dino_image', 
                    'enable_cfg': 0, 
                    'loss': 0, 
                    'special_token_loss': 0,
                    'special_token_label': None,
                }
            )

        if need_vit:
            data['sequence_plan'].append(
                {
                    'type': 'vit_image',
                    'enable_cfg': 0, 
                    'loss': 0,
                    'special_token_loss': 0,
                    'special_token_label': None,
                },
            )

            vit_image_tensor, image_grid_thw = self.vit_transform([image], img_num=1) 
            data['num_tokens'] += vit_image_tensor.shape[0] // 4 
            data['image_tensor_list'].append(vit_image_tensor)
            data['image_grid_thw_list'].append(image_grid_thw[0])

        return data

    def _add_video(self, data, frames, frame_indexes, need_loss, need_vae, enable_cfg=True): 
        assert int(need_loss) + int(need_vae) == 1

        if need_loss:
            for idx, (image, frame_idx) in enumerate(zip(frames, frame_indexes)):
                current_sequence_plan = {
                    'type': 'vae_image', 
                    'enable_cfg': 0, 
                    'loss': 1, 
                    'special_token_loss': 0,
                    'special_token_label': None,
                    'split_start': idx == 0,
                    'split_end': idx == len(frames) - 1,
                }
                if idx < len(frame_indexes) - 1:
                    current_sequence_plan['frame_delta'] = frame_indexes[idx + 1] - frame_idx
                data['sequence_plan'].append(current_sequence_plan)
                image_tensor = self.transform(image)
                height, width = image_tensor.shape[1:]
                data['image_tensor_list'].append(image_tensor)
                data['num_tokens'] += width * height // self.transform.stride ** 2

        elif need_vae:
            for idx, (image, frame_idx) in enumerate(zip(frames, frame_indexes)):
                current_sequence_plan = {
                    'type': 'vae_image', 
                    'enable_cfg': int(enable_cfg), 
                    'loss': 0, 
                    'special_token_loss': 0,
                    'special_token_label': None,
                    'split_start': idx == 0,
                    'split_end': idx == len(frames) - 1,
                }
                if idx < len(frame_indexes) - 1:
                    current_sequence_plan['frame_delta'] = frame_indexes[idx + 1] - frame_idx
                data['sequence_plan'].append(current_sequence_plan)
                image_tensor = self.transform(image)
                height, width = image_tensor.shape[1:]
                data['image_tensor_list'].append(image_tensor)
                data['num_tokens'] += width * height // self.transform.stride ** 2

        return data

    def parse_row(self, row):
        question = row["question"]
        answer = row["answer"]

        images_list = []
        depths_list = []
        poses_list = []
        view_infos = []
        depth_intrinsic_list = []
        intrinsic_list = []


        shuffle_seq_views = False 
        img_per_seq = self.random_image_num = self.frame_num

        data_scene_name = row['scene_name']
        dataset_name = row['dataset_name']
        
        rng = self._rng

        if 'spar' in dataset_name:

            num_imgs = len(row['image_list']) 
            img_per_seq = num_imgs

            images_list = list(row['image_list'])
            depths_list = list(row['depth_list'])
            poses_list = list(row['poses'])
            this_scene = row['scene_name']
            assert len(images_list) == len(depths_list) == len(poses_list)

            current_len = len(images_list)

            for idx, image in enumerate(images_list):
                depth_intrinsic = row['depth_intrinsic']
                intrinsic = row['intrinsic']
                view_infos.append(f'{data_scene_name}/{dataset_name}/{this_scene}/{str(idx)}')
                depth_intrinsic_list.append(depth_intrinsic)
                intrinsic_list.append(intrinsic)
                

        dino_meta = {}
        dino_meta['scene_name'] = data_scene_name

        if data_scene_name == 'scannet' or data_scene_name == 'structured3d':
            intric_list = depth_intrinsic_list
        else:
            intric_list = intrinsic_list

        image_list_copy = deepcopy(images_list)

        vit_images_list = []
        if 'spar' in dataset_name:
            try:
                metadata = row['metadata']
                if not isinstance(metadata, dict):
                    safe_context = {
                        "array": np.array,
                        "object": object,
                        "None": None,
                        "null": None,  # <-- Add this line to map 'null' to Python's 'None'
                    }
                    metadata = eval(metadata, safe_context)
            
                image_types = {
                    "obj_spatial_relation_oo",
                    "depth_prediction_oc",
                    "depth_prediction_oo",
                    "distance_prediction_oc",
                    "distance_prediction_oo",
                    "distance_infer_center_oc",
                    "distance_infer_center_oo",
                    "spatial_volume_infer",
                    "spatial_imagination_oc",
                    "spatial_imagination_oo",
                }
                images_type = {
                    "position_matching",
                    "view_change_infer",
                    "depth_prediction_oc_mv",
                    "depth_prediction_oo_mv",
                    "distance_prediction_oc_mv",
                    "distance_prediction_oo_mv",
                    "obj_spatial_relation_oc_mv",
                    "obj_spatial_relation_oo_mv",
                    "distance_infer_center_oc_mv",
                    "distance_infer_center_oo_mv",
                    "spatial_imagination_oc_mv",
                    "spatial_imagination_oo_mv",
                    "spatial_imagination_map_mv",
                    "camera_motion_infer",
                    "distance_prediction_oo_video",
                    "distance_infer_center_oo_video",
                    "spatial_imagination_oo_video",
                    "spatial_imagination_oc_video",
                    "spatial_imagination_oc_video_hard",
                    "spatial_imagination_oo_video_hard",
                    "obj_frame_locate",
                    "appearance_order",
                    "room_size",
                    "obj_count",
                    "nav",
                }
                if metadata['type'] in image_types:
                    for i, img_path in enumerate(images_list):
                        try:
                            image = Image.open(img_path).convert("RGB") 
                            self.draw_image(image, metadata)
                            vit_images_list.append(image)
                        except Exception as e:
                            error_print_once(e, f"| img_path={img_path} | data_item_id={metadata.get('id', 'unk')}")
                elif metadata['type'] in images_type:
                    images = [Image.open(p).convert('RGB') for p in images_list]
                    self.draw_image(images, metadata)
                    vit_images_list = images
                    
                raw_images = vit_images_list
            except Exception as e:
                print('e', e)
                print('metadata', metadata)
                print('row:', row)
                raw_images = [
                    pil_img2rgb(Image.open(image))
                    for image in images_list
                ]
        else:
            raw_images = [
                    pil_img2rgb(Image.open(image))
                    for image in images_list
                ]
        raw_images_dino = [
            np.array(Image.open(image).convert("RGB"))
                for image in image_list_copy
        ]

        data = self._init_data()
        data['img_per_seq'] = len(images_list)
        data['image_paths'] = images_list

        text_with_images = '<dino_image>'*len(raw_images_dino)+'<vit_image>'*len(raw_images)+question
        split_list = apply_template_qwenvl2_reconThenUnd(text_with_images, answer)

        for item in split_list:
            try:
                if item['type'] == 'text':
                    data = self._add_text(data, item["value"], need_loss=item['loss'])
                elif item['type'] == 'dino':
                    depth, depths_list = self.pop_first(depths_list)
                    pose, poses_list = self.pop_first(poses_list)
                    image, raw_images_dino = self.pop_first(raw_images_dino)
                    this_view_info, view_infos = self.pop_first(view_infos)
        
                    intric, intric_list = self.pop_first(intric_list)
                    dino_meta['intri'] = np.vstack(intric).reshape((4, 4))


                    dino_meta['depth'] = depth
                    dino_meta['pose'] = np.vstack(pose).reshape((4, 4))

                    data = self._add_image(
                        data, 
                        image,
                        dino_meta=dino_meta,
                        need_loss=False, 
                        need_dino=True, 
                        need_vit=False, 
                        rng=rng,
                        view_info=this_view_info,
                    )
                elif item['type'] == 'vit':
                    image, raw_images = self.pop_first(raw_images)
                    data = self._add_image(
                            data, 
                            image,
                            dino_meta=dino_meta,
                            need_loss=False, 
                            need_dino=False, 
                            need_vit=True, 
                        )
            except AssertionError as e:
                print(e, 'skipping')
                return [] 

        return data
