import json 
from data.dataset_utils_vggt import *
from pi3.utils.geometry import depthmap_to_absolute_camera_coordinates
import torch 

path = 'path of your processed annotation (a jsonl file)'

with open(path, 'r') as f: 
    data = [json.loads(l) for l in f.readlines()]
data_ind = 0
data_item = data[data_ind]
images = []
points = []
data_scene_name = data_item['scene_name']
this_scene = data_item['seq_name']

#########################################################
rng = np.random.default_rng(0)
sampling_strategy = None
class self():
    frame_num = 10
    random_sample_thres = 0.1
    resolution=[518, 518]
    _rng = rng
    ceph_read = True   
    aug_focal = 0 #0.9
    aug_crop = 0 #16 ###aug_crop
    
images = []
depths = []
cam_points = []
world_points = []
point_masks = []
extrinsics = []
intrinsics = []
view_infos = []
########################################## 
scannetpp_dir ='input your data dir'

scene = this_scene
scene_path = osp.join(scannetpp_dir, scene)

metadata_path = os.path.join(scene_path, "scene_metadata.npz")
metadata = load_npz(metadata_path, ceph_read=False)
meta_intrinsics = metadata["intrinsics"]
trajectories = metadata["trajectories"]
meta_images = metadata["images"]

scene_id = scene_path.split("/")[-1].split(".")[0]
scene_frames = [
    {
        "file_path": os.path.join(scene_path, "images", meta_images[i].split(".")[0] + ".jpg"),
        "depth_path": os.path.join(scene_path, "depth", meta_images[i].split(".")[0] + ".png"),
        "intrinsics": meta_intrinsics[i],
        "extrinsics": trajectories[i],
    }
    for i in range(len(trajectories))
]
scene_frames.sort(key=lambda x: x["file_path"]) 
meta_extrinsics = np.array([frame["extrinsics"] for frame in scene_frames])
meta_intrinsics = np.array([frame["intrinsics"] for frame in scene_frames])
rgb_paths = np.array([frame["file_path"] for frame in scene_frames])
depth_paths = np.array([frame["depth_path"] for frame in scene_frames])
num_imgs = len(meta_extrinsics)
        
valid_idxs = [i for i in range(num_imgs)]

if self.frame_num > 20 and rng.random() < self.random_sample_thres and num_imgs < 16 * self.frame_num:
    all_keys = valid_idxs
    should_replace = len(all_keys) < self.frame_num
    idxs = list(rng.choice(all_keys, size=self.frame_num, replace=should_replace))
else:
    idxs = [rng.integers(0, num_imgs)]

    scanetpp_max_distance = 240
    max_distance = int(scanetpp_max_distance/ 8 * self.frame_num)
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
    
for idx in idxs:
        
    image_filepath = rgb_paths[idx]
    depth_filepath = depth_paths[idx]
    rgb_image = np.array(Image.open(image_filepath))
    with Image.open(depth_filepath) as depth_img:
        depthmap = np.array(depth_img).astype(np.int32)

    depthmap = depthmap.astype(np.float32) / 1000
    depthmap[~np.isfinite(depthmap)] = 0
    
    intrinsic = meta_intrinsics[idx]
    camera_pose = meta_extrinsics[idx]

    rgb_image, depthmap, intrinsic_ = crop_resize_if_necessary(self,
        rgb_image, depthmap, intrinsic.copy(), self.resolution, rng=rng, info=image_filepath)

    images.append(rgb_image)
    depths.append(depthmap.astype(np.float32))
    extrinsics.append(camera_pose.astype(np.float32))
    intrinsics.append(intrinsic_.astype(np.float32))
    view_infos.append(f'{data_scene_name}/{scene}/{str(idx)}')

new_depths = []
for v,(img,depthmap,camera_pose, camera_intrinsics, view_info) in enumerate(zip(images,depths,extrinsics,intrinsics, view_infos)):

    width, height = img.size

    assert np.isfinite(camera_pose).all(), f'NaN in camera pose for view {view_info}'
    assert np.isfinite(depthmap).all(), f'NaN in depthmap for view {view_info}'

    scene_label = view_info.split('/')[0]
    if scene_label in ['co3dv2', 'wildrgbd', 'blendedmvs']: 
        z_far = 0
    elif scene_label in ['tartanair', 'gtasfm', 'matrixcity', "taskonomy", 'hypersim', 'nav_20w', 'vkitti']:
        z_far = 0
    else:
        z_far = 80 

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

predictions = {}
predictions['world_points'] = world_points
predictions['point_masks'] = point_masks
predictions['images'] = images
predictions['view_infos'] = view_infos

save_ply_visualization_cpu_batch(predictions, f'{str(data_ind)}_norandom', f'QuickVis_debug{data_scene_name}', gt_only=True)

