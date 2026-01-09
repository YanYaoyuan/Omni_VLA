import random
import json

import numpy as np
import torch
import torchvision.transforms as transforms
from typing import Any, Dict, List, Mapping, Optional, Sequence


from .data_utils import (
    get_flattened_position_ids_interpolate,
    get_rope_index_image_3D,
    get_rope_index_image_3D_dino,
    get_flattened_position_ids_extrapolate, 
    len2weight,
    patchify, 
    prepare_attention_mask_per_sample, 
)
from .dataset_info import DATASET_INFO, DATASET_REGISTRY
from .transforms import ImageTransform
from .transforms_vggt import DinoImageTransform
from .transforms import QwenVL2ImageTransform
from .video_utils import FrameSampler
from .augmentation_vggt import get_image_augmentation

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class DataConfig:
    def __init__(
        self, 
        grouped_datasets, 
        text_cond_dropout_prob=0.1,
        vit_cond_dropout_prob=0.4,
        dino_cond_dropout_prob=0.4,
        max_latent_size=32,
        vit_patch_size=14,
        dino_patch_size=14,
        vit_max_num_patch_per_side=70,
        dino_max_num_patch_per_side=37,
    ):
        self.grouped_datasets = grouped_datasets
        self.text_cond_dropout_prob = text_cond_dropout_prob
        self.vit_cond_dropout_prob = vit_cond_dropout_prob
        self.vit_patch_size = vit_patch_size
        self.dino_cond_dropout_prob = dino_cond_dropout_prob
        self.dino_patch_size = dino_patch_size
        self.vit_max_num_patch_per_side = vit_max_num_patch_per_side
        self.dino_max_num_patch_per_side = dino_max_num_patch_per_side
        self.max_latent_size = max_latent_size


class PackedDataset(torch.utils.data.IterableDataset):
    def __init__(
        self, 
        data_config, 
        tokenizer, 
        special_tokens,
        local_rank, 
        world_size, 
        num_workers,
        expected_num_tokens=32768, 
        max_num_tokens_per_sample=16384,
        max_num_tokens=36864,
        prefer_buffer_before=16384,
        max_buffer_size=50,
        interpolate_pos=False,
        use_flex=False,
        data_status=None,
    ):
        super().__init__()
        self.expected_num_tokens = expected_num_tokens
        self.max_num_tokens_per_sample = max_num_tokens_per_sample
        self.prefer_buffer_before = prefer_buffer_before
        self.max_num_tokens = max_num_tokens
        self.max_buffer_size = max_buffer_size
        self.tokenizer = tokenizer
        self.local_rank = local_rank
        self.world_size = world_size
        self.num_workers = num_workers
        self.use_flex = use_flex
        self.step_counter = 0
        for k, v in special_tokens.items():
            setattr(self, k, v)
        
        #added for aug 
        self.cojitter = True  #common_config.augs.cojitter
        # Probability of using shared jitter vs. frame-specific jitter
        self.cojitter_ratio = 0.3  #common_config.augs.cojitter_ratio
        # Initialize image augmentations (color jitter, grayscale, gaussian blur)
        self.image_aug = get_image_augmentation(
            gray_scale=True,
            gau_blur=False,
            color_jitter=None,
        )
  
        self.resnet_normalize = transforms.Normalize(mean=_RESNET_MEAN, std=_RESNET_STD)

        grouped_names, grouped_datasets, is_mandatory, grouped_weights = self.build_datasets(
            data_config.grouped_datasets, data_status
        )
        self.grouped_datasets = grouped_datasets
        self.dataset_iters = [(iter(dataset), grouped_name, dataset) for (dataset, grouped_name) in zip(grouped_datasets, grouped_names)]          
        self.is_mandatory = is_mandatory
        self.grouped_weights = grouped_weights
        self.data_config = data_config
        self.interpolate_pos = interpolate_pos
        if self.interpolate_pos:
            self.get_flattened_position_ids = get_flattened_position_ids_interpolate
        else:
            self.get_flattened_position_ids = get_flattened_position_ids_extrapolate
        
        self.aspect_ratio_range = [0.5, 1.2] 
        self.image_num_range = [2, 24] 
        self.image_num_weights = {num_images: 1.0 for num_images in range(self.image_num_range[0], self.image_num_range[1]+1)}

        self.possible_nums = np.array([n for n in self.image_num_weights.keys()
                                    if self.image_num_range[0] <= n <= self.image_num_range[1]])

        weights = [self.image_num_weights[n] for n in self.possible_nums]
        self.normalized_weights = np.array(weights) / sum(weights)
        self.base_and_epoch_seed = 42
        self.step_counter = 0

    def build_datasets(self, datasets_metainfo, data_status):
        datasets = []
        is_mandatory = []
        grouped_weights = []
        grouped_names = []
        for grouped_dataset_name, dataset_args in datasets_metainfo.items():
            is_mandatory.append(dataset_args.pop('is_mandatory', False))
            grouped_weights.append(dataset_args.pop('weight', 0.0))

            if 'frame_sampler_args' in dataset_args.keys():
                frame_sampler = FrameSampler(**dataset_args.pop('frame_sampler_args'))
                dataset_args['frame_sampler'] = frame_sampler
            if 'image_transform_args' in dataset_args.keys():
                transform = ImageTransform(**dataset_args.pop('image_transform_args'))
                dataset_args['transform'] = transform
            if 'vit_image_transform_args' in dataset_args.keys():
                vit_transform = QwenVL2ImageTransform(**dataset_args.pop('vit_image_transform_args'))
                dataset_args['vit_transform'] = vit_transform
            if 'dino_image_transform_args' in dataset_args.keys():
                dino_transform = DinoImageTransform(**dataset_args.pop('dino_image_transform_args'))
                dataset_args['dino_transform'] = dino_transform

            assert 'dataset_names' in dataset_args.keys()
            dataset_names = dataset_args.pop('dataset_names')
            dataset_args['data_dir_list'] = []
            for item in dataset_names:
                if self.local_rank == 0:
                    print(f'Preparing Dataset {grouped_dataset_name}/{item}')
                meta_info = DATASET_INFO[grouped_dataset_name][item]
                dataset_args['data_dir_list'].append(meta_info['data_dir'])

                if "parquet_info_path" in meta_info.keys():
                    if 'parquet_info' not in dataset_args.keys():
                        dataset_args['parquet_info'] = {}
                    with open(meta_info['parquet_info_path'], 'r') as f:
                        parquet_info = json.load(f)
                    dataset_args['parquet_info'].update(parquet_info)

                if 'json_dir' in meta_info.keys():
                    # parquet/tar with json
                    if 'json_dir_list' not in dataset_args.keys():
                        dataset_args['json_dir_list'] = [meta_info['json_dir']]
                    else:
                        dataset_args['json_dir_list'].append(meta_info['json_dir'])

                if 'jsonl_path' in meta_info.keys():
                    # jsonl with jpeg
                    if 'jsonl_path_list' not in dataset_args.keys():
                        dataset_args['jsonl_path_list'] = [meta_info['jsonl_path']]
                    else:
                        dataset_args['jsonl_path_list'].append(meta_info['jsonl_path'])

            resume_data_status = dataset_args.pop('resume_data_status', True)
            if data_status is not None and grouped_dataset_name in data_status.keys() and resume_data_status:
                data_status_per_group = data_status[grouped_dataset_name]
            else:
                data_status_per_group = None
            dataset = DATASET_REGISTRY[grouped_dataset_name](
                dataset_name=grouped_dataset_name,
                tokenizer=self.tokenizer,
                local_rank=self.local_rank,
                world_size=self.world_size,
                num_workers=self.num_workers,
                data_status=data_status_per_group,
                **dataset_args
            )
            datasets.append(dataset)
            grouped_names.append(grouped_dataset_name)
     
        return grouped_names, datasets, is_mandatory, grouped_weights

    def set_epoch(self, seed):
        for dataset in self.grouped_datasets:
            dataset.set_epoch(seed)
        self.base_and_epoch_seed = seed

    def set_sequence_status(self):
        sequence_status = dict(
            curr                        = 0,
            sample_lens                 = list(),
            packed_position_ids         = list(),
            nested_attention_masks      = list(),
            split_lens                  = list(),
            attn_modes                  = list(),
            packed_text_ids             = list(), 
            packed_text_indexes         = list(),
            packed_label_ids            = list(),
            ce_loss_indexes             = list(),
            ce_loss_weights             = list(),


            dino_token_seqlens          = list(),
            packed_dino_token_indexes   = list(), 
            packed_depths               = list(),
            packed_extrinsics           = list(),
            packed_intrinsics           = list(),
            packed_cam_points           = list(),
            packed_world_points         = list(),
            packed_point_masks          = list(),
            packed_view_infos           = list(),
            packed_image_paths           = list(),
            packed_dino_image_tensor_list    = list(),
            packed_image_grid_thw       = list(),

            packed_vit_tokens           = list(), 
            packed_vit_images           = list(), 
            vit_token_seqlens           = list(),
            packed_vit_token_indexes    = list(), 
            img_per_seq_lens            = list(), 
        )
        return sequence_status

    def to_tensor(self, sequence_status):
        data = dict(
            sequence_length=sum(sequence_status['sample_lens']),
            sample_lens=sequence_status['sample_lens'],
            packed_text_ids=torch.tensor(sequence_status['packed_text_ids']),
            packed_text_indexes=torch.tensor(sequence_status['packed_text_indexes']),
            packed_position_ids=torch.cat(sequence_status['packed_position_ids'], dim=1),
        )
        if not self.use_flex:
            data['nested_attention_masks'] = sequence_status['nested_attention_masks']
        else:
            sequence_len = data['sequence_length']
            pad_len = self.max_num_tokens - sequence_len #### this is fixed only postive num
            data['split_lens'] = sequence_status['split_lens'] + [pad_len]
            data['attn_modes'] = sequence_status['attn_modes'] + ['causal']
            data['sample_lens'] += [pad_len]
        
        if len(sequence_status['packed_dino_image_tensor_list']) > 0: 

            data['packed_dino_token_indexes'] = torch.tensor(sequence_status['packed_dino_token_indexes'])
            data['dino_token_seqlens'] = torch.tensor(sequence_status['dino_token_seqlens'])


            packed_dino_image_tensors = torch.from_numpy(np.stack(sequence_status["packed_dino_image_tensor_list"]).astype(np.float32)).contiguous()
            packed_dino_image_tensors = packed_dino_image_tensors.permute(0,3,1,2).to(torch.get_default_dtype()).div(255)

            if self.image_aug is not None:
                if self.cojitter and random.random() > self.cojitter_ratio:
                    # Apply the same color jittering transformation to all frames
                    packed_dino_image_tensors = self.image_aug(packed_dino_image_tensors)
                else:
                    # Apply different color jittering to each frame individually
                    for aug_img_idx in range(len(packed_dino_image_tensors)):
                        packed_dino_image_tensors[aug_img_idx] = self.image_aug(packed_dino_image_tensors[aug_img_idx])
            
            packed_dino_image_tensors = packed_dino_image_tensors.contiguous()

            depths = torch.from_numpy(np.stack(sequence_status["packed_depths"]).astype(np.float32)).to(torch.float32)
            extrinsics = torch.from_numpy(np.stack(sequence_status["packed_extrinsics"]).astype(np.float32)).to(torch.float32)
            intrinsics = torch.from_numpy(np.stack(sequence_status["packed_intrinsics"]).astype(np.float32)).to(torch.float32)
            world_points = torch.from_numpy(np.stack(sequence_status["packed_world_points"]).astype(np.float32)).to(torch.float32)
            point_masks = torch.from_numpy(np.stack(sequence_status["packed_point_masks"])) # Mask indicating valid depths / world points / cam points per frame

            data["packed_depths"] = depths             
            data["packed_extrinsics"] =  extrinsics
            data["packed_intrinsics"] = intrinsics
            # data["packed_cam_points"] = cam_points
            data["packed_world_points"] = world_points
            data["packed_point_masks"] = point_masks
            # packed_dino_image_tensors = torch.stack(sequence_status["packed_dino_image_tensor_list"], dim=0)

            packed_dino_image_tensors = self.resnet_normalize(packed_dino_image_tensors)
            data["packed_dino_image_tensor_list"]  = packed_dino_image_tensors
            data['img_per_seq_lens'] = sequence_status['img_per_seq_lens']
            data['packed_view_infos'] = sequence_status["packed_view_infos"]
            data['packed_image_paths'] = sequence_status["packed_image_paths"]
            

        if len(sequence_status['packed_vit_images']) > 0:
            data['packed_vit_images'] =  torch.stack(sequence_status['packed_vit_images'], dim=0)
            data['packed_vit_token_indexes'] = torch.tensor(sequence_status['packed_vit_token_indexes'])
            data['vit_token_seqlens'] = torch.tensor(sequence_status['vit_token_seqlens'])
            data['packed_image_grid_thw'] = torch.stack(sequence_status['packed_image_grid_thw'], dim=0)

        # if the model is required to perform text generation
        if len(sequence_status['packed_label_ids']) > 0:
            data['packed_label_ids'] = torch.tensor(sequence_status['packed_label_ids'])
            data['ce_loss_indexes'] = torch.tensor(sequence_status['ce_loss_indexes'])
            data['ce_loss_weights'] = torch.tensor(sequence_status['ce_loss_weights'])

        return data

    def __iter__(self):
        total_weights = sum(self.grouped_weights)
        assert total_weights > 0.0
        group_cumprobs = [sum(self.grouped_weights[:i + 1]) / total_weights 
                        for i in range(len(self.grouped_weights))]
        sequence_status = self.set_sequence_status()
        batch_data_indexes = []

        while True:
            self.step_counter += 1
            step_seed = self.base_and_epoch_seed + self.step_counter
            step_rng = random.Random(step_seed)
            
            random_image_num = int(step_rng.choices(
                self.possible_nums, 
                weights=self.normalized_weights,
                k=1
            )[0])
            random_aspect_ratio = round(
                step_rng.uniform(self.aspect_ratio_range[0], self.aspect_ratio_range[1]), 
                2
            )      

            # Ensure at least one sample from each group
            if sequence_status['curr'] == 0:
                for group_index, group_iter_names in enumerate(self.dataset_iters):
                    group_iter = group_iter_names[0]
                    group_name = group_iter_names[1]
                    group_dataset = group_iter_names[2]  
                    if self.is_mandatory[group_index]: #grouped means recon group.  
                        while True:
                            if group_name == "recon":
                                group_dataset.set_random_image_num(random_image_num)
                                group_dataset.set_random_aspect_ratio(random_aspect_ratio)
                                group_dataset.set_step_rng(step_seed)
                                sample = next(group_iter)
                            else:
                                sample = next(group_iter)
                                
                            if sample is None:
                                continue
                            num_tokens = sample['num_tokens'] + 2 * len(sample['sequence_plan']) 
                            if num_tokens < self.max_num_tokens_per_sample:
                                sequence_status = self.pack_sequence(sample, sequence_status)
                                batch_data_indexes.append(sample['data_indexes'])
                                break
                            else:
                                print(f"skip a sample with length {num_tokens}")
                                continue

            n = random.random()
            group_index = 0
            for i, cumprob in enumerate(group_cumprobs):
                if n < cumprob:
                    group_index = i
                    break
            sample = next(self.dataset_iters[group_index][0])

            num_tokens = sample['num_tokens'] + 2 * len(sample['sequence_plan'])
            if num_tokens > self.max_num_tokens_per_sample:
                print(f"skip a sample with length {num_tokens}")
                continue

            if sequence_status['curr'] + num_tokens > self.max_num_tokens:
                data = self.to_tensor(sequence_status)
                data['batch_data_indexes'] = batch_data_indexes
                yield data
                sequence_status = self.set_sequence_status()
                batch_data_indexes = []
                continue

            sequence_status = self.pack_sequence(sample, sequence_status)
            batch_data_indexes.append(sample['data_indexes'])

            if sequence_status['curr'] >= self.expected_num_tokens:
                print(f"Yielding data exceed expected_num_tokens with length {sum(sequence_status['sample_lens'])}, num_tokens is {num_tokens}")
                data = self.to_tensor(sequence_status)
                data['batch_data_indexes'] = batch_data_indexes
                yield data
                sequence_status = self.set_sequence_status()
                batch_data_indexes = []

    def pack_sequence(self, sample, sequence_status):
        if 'image_tensor_list' in sample:
            image_tensor_list = sample['image_tensor_list']
        if 'image_grid_thw_list' in sample: 
            image_grid_thw_list = sample['image_grid_thw_list']
        text_ids_list = sample['text_ids_list']
        sequence_plan = sample['sequence_plan']

        img_per_seq = 0
        if 'depths' in sample:
            depth_array =sample['depths']
        if 'extrinsics' in sample:
            extrinsics_array =sample['extrinsics']
        if 'intrinsics' in sample:
            intrinsics_array =sample['intrinsics']
        if 'world_points' in sample:
            world_points_array =sample['world_points']
        if 'point_masks' in sample:
            point_masks_array =sample['point_masks']
        if 'view_infos' in sample:
            view_infos =sample['view_infos']
        if 'image_paths' in sample:
            image_paths =sample['image_paths']
        if 'img_per_seq' in sample: 
            img_per_seq = sample['img_per_seq']
        if 'dino_image_tensor_list' in sample: 
            dino_image_tensor_list = sample['dino_image_tensor_list']
        if 'dino_images' in sample: 
            dino_images = sample['dino_images']
        if 'dino_thw' in sample: 
            dino_thw = sample['dino_thw']

        split_lens, attn_modes = list(), list()
        curr = sequence_status['curr']
        curr_rope_id = 0
        sample_lens = 0
        vit_cnt = 0
        dino_cnt = 0


        for item in sequence_plan:
            split_start = item.get('split_start', True)
            if split_start:
                curr_split_len = 0

            if item['type'] == 'text':
                text_ids = text_ids_list.pop(0)
                if item['enable_cfg'] == 1 and random.random() < self.data_config.text_cond_dropout_prob:
                    continue

                shifted_text_ids = text_ids
                sequence_status['packed_text_ids'].extend(shifted_text_ids)
                sequence_status['packed_text_indexes'].extend(range(curr, curr + len(shifted_text_ids)))

                # \n text_token  <-> text_token <|im_end|>
                if item['loss'] == 1:
                    sequence_status['ce_loss_indexes'].extend(range(curr, curr + len(shifted_text_ids)))
                    sequence_status['ce_loss_weights'].extend(
                        [len2weight(len(shifted_text_ids))] * len(shifted_text_ids)
                    )
                    sequence_status['packed_label_ids'].extend(text_ids[1:] + [self.eos_token_id])
                curr += len(shifted_text_ids)
                curr_split_len += len(shifted_text_ids)

                if item['loss'] == 1:
                    sequence_status['packed_text_ids'].append(self.eos_token_id)
                    sequence_status['packed_text_indexes'].append(curr)
                    curr += 1
                    curr_split_len += 1

                attn_modes.append("causal")
                pos_ids = torch.tensor(range(curr_rope_id, curr_rope_id + curr_split_len), dtype=torch.long).expand(3, -1),
                sequence_status['packed_position_ids'].extend(pos_ids)
                curr_rope_id += curr_split_len

            elif item['type'] == 'vit_image':
                vit_cnt += 1


                image_tensor = image_tensor_list.pop(0)
                image_grid_thw = image_grid_thw_list.pop(0)

                if item['enable_cfg'] == 1 and random.random() < self.data_config.vit_cond_dropout_prob:
                    curr_rope_id += 1
                    continue

                sequence_status['packed_text_ids'].append(self.start_of_image)
                sequence_status['packed_text_indexes'].append(curr)
                curr += 1
                curr_split_len += 1

                pos_tensor = torch.full((1,), curr_rope_id, dtype=torch.long)
                sequence_status['packed_position_ids'].extend([pos_tensor.expand(3, 1)])
                curr_rope_id += 1


                num_img_tokens =  image_tensor.shape[0] // 4
                sequence_status['packed_vit_token_indexes'].extend(range(curr, curr + num_img_tokens))
                curr += num_img_tokens
                curr_split_len += num_img_tokens

                sequence_status['packed_vit_images'].append(image_tensor)
                sequence_status['vit_token_seqlens'].append(num_img_tokens)
                sequence_status['packed_image_grid_thw'].append(image_grid_thw)

                postions_ids_from_vit_for_rope, rope_deltas = get_rope_index_image_3D(
                    image_grid_thw,
                    curr_rope_id,
                    device=image_tensor.device
                )

                sequence_status['packed_position_ids'].extend([postions_ids_from_vit_for_rope])
                curr_rope_id += rope_deltas + 1

                sequence_status['packed_text_ids'].append(self.end_of_image)
                sequence_status['packed_text_indexes'].append(curr)
                if item['special_token_loss'] == 1: # <|endofimage|> may have loss
                    sequence_status['ce_loss_indexes'].append(curr)
                    sequence_status['ce_loss_weights'].append(1.0)
                    sequence_status['packed_label_ids'].append(item['special_token_label'])
                curr += 1
                curr_split_len += 1

                pos_tensor = torch.full((1,), curr_rope_id, dtype=torch.long)
                sequence_status['packed_position_ids'].extend([pos_tensor.expand(3, 1)])
                curr_rope_id += 1

                attn_modes.append("full") # changed to noise
       

            elif item['type'] == 'dino_image':
                dino_cnt+=1
                use_registers = False
                use_camera_token = False 
                
                dino_image_thw = dino_thw.pop(0)
                image_tensor = dino_image_tensor_list.pop(0)
                dino_image = dino_images.pop(0)
                depths_np = depth_array.pop(0)
                extrinsics_np = extrinsics_array.pop(0)
                intrinsics_np = intrinsics_array.pop(0)
                world_points_np  = world_points_array.pop(0)
                point_masks_np = point_masks_array.pop(0)        

                if 'view_infos' in sample:
                    view_info_str = view_infos.pop(0)
                else:
                    view_info_str = ''
                    
                if 'image_paths' in sample:
                    image_path_str = image_paths.pop(0)
                else:
                    image_path_str = ''
                
                sequence_status["packed_depths"].append(depths_np)               
                sequence_status["packed_extrinsics"].append(extrinsics_np)  
                sequence_status["packed_intrinsics"].append(intrinsics_np)  
                sequence_status["packed_world_points"].append(world_points_np)  
                sequence_status["packed_point_masks"].append(point_masks_np)  
                sequence_status["packed_view_infos"].append(view_info_str)  
                sequence_status["packed_image_paths"].append(image_path_str)  
                sequence_status["packed_dino_image_tensor_list"].append(dino_image)  

                sequence_status['packed_text_ids'].append(self.start_of_image)
                sequence_status['packed_text_indexes'].append(curr)
                curr += 1
                curr_split_len += 1
                
                pos_tensor = torch.full((1,), curr_rope_id, dtype=torch.long)
                sequence_status['packed_position_ids'].extend([pos_tensor.expand(3, 1)])
                curr_rope_id += 1

                if use_camera_token:
                    pos_tensor = torch.full((1,), curr_rope_id, dtype=torch.long)
                    sequence_status['packed_position_ids'].extend([pos_tensor.expand(3, 1)])
                    curr_rope_id += 1 

                if use_registers:
                    for _ in range(4):
                        pos_tensor = torch.full((1,), curr_rope_id, dtype=torch.long)
                        sequence_status['packed_position_ids'].extend([pos_tensor.expand(3, 1)])
                        curr_rope_id += 1

                # preprocess image
                dino_tokens = patchify(image_tensor, self.data_config.dino_patch_size)
                num_img_tokens = dino_tokens.shape[0]
                # note: +1 is for one camera token 
                
                if use_registers:
                    sequence_status['packed_dino_token_indexes'].extend(range(curr, curr + num_img_tokens +1 + 4))
                    curr += num_img_tokens + 1 + 4
                    curr_split_len += num_img_tokens + 1 +4 
                elif use_camera_token:
                    sequence_status['packed_dino_token_indexes'].extend(range(curr, curr + num_img_tokens + 1))
                    curr += num_img_tokens + 1 
                    curr_split_len += num_img_tokens + 1
                else:
                    sequence_status['packed_dino_token_indexes'].extend(range(curr, curr + num_img_tokens ))
                    curr += num_img_tokens 
                    curr_split_len += num_img_tokens 

                # sequence_status['packed_dino_tokens'].append(dino_tokens)
                sequence_status['dino_token_seqlens'].append(num_img_tokens)

                postions_ids_from_dino_for_rope, rope_deltas = get_rope_index_image_3D_dino(
                    dino_image_thw,
                    curr_rope_id,
                    device=image_tensor.device
                )
                sequence_status['packed_position_ids'].extend([postions_ids_from_dino_for_rope])
                curr_rope_id += rope_deltas + 1

                # add a <|endofimage|> token
                sequence_status['packed_text_ids'].append(self.end_of_image)
                sequence_status['packed_text_indexes'].append(curr)
                if item['special_token_loss'] == 1: # <|endofimage|> may have loss
                    sequence_status['ce_loss_indexes'].append(curr)
                    sequence_status['ce_loss_weights'].append(1.0)
                    sequence_status['packed_label_ids'].append(item['special_token_label'])
                curr += 1
                curr_split_len += 1

                # add a <|endofimage|> token 3d rope pos
                pos_tensor = torch.full((1,), curr_rope_id, dtype=torch.long)
                sequence_status['packed_position_ids'].extend([pos_tensor.expand(3, 1)])
                curr_rope_id += 1

                # update sequence status
                attn_modes.append("full")

            if item.get('split_end', True):
                split_lens.append(curr_split_len)
                sample_lens += curr_split_len

        sequence_status['curr'] = curr
        sequence_status['sample_lens'].append(sample_lens)
        sequence_status["img_per_seq_lens"].append(img_per_seq)
        # prepare attention mask
        if not self.use_flex:
            sequence_status['nested_attention_masks'].append(
                prepare_attention_mask_per_sample(split_lens, attn_modes)
            )
        else:
            sequence_status['split_lens'].extend(split_lens)
            sequence_status['attn_modes'].extend(attn_modes)

        return sequence_status

class SimpleCustomBatch:
    def __init__(self, batch):
        data = batch[0]
        self.batch_data_indexes = data['batch_data_indexes']
        self.sequence_length = data["sequence_length"]
        self.sample_lens = data["sample_lens"]
        self.packed_text_ids = data["packed_text_ids"]
        self.packed_text_indexes = data["packed_text_indexes"]
        self.packed_position_ids = data["packed_position_ids"]

        self.use_flex = "nested_attention_masks" not in data.keys()

        if self.use_flex:
            self.split_lens = data["split_lens"]
            self.attn_modes = data["attn_modes"]
        else:
            self.nested_attention_masks = data["nested_attention_masks"]

        if "packed_dino_image_tensor_list" in data.keys():

            self.packed_dino_token_indexes = data['packed_dino_token_indexes']
            self.dino_token_seqlens = data["dino_token_seqlens"]

            self.packed_depths =  data["packed_depths"]          
            self.packed_extrinsics =  data["packed_extrinsics"] 
            self.packed_intrinsics =  data["packed_intrinsics"]
            self.packed_world_points =  data["packed_world_points"]
            self.packed_point_masks = data["packed_point_masks"] 
            self.packed_dino_image_tensor_list = data["packed_dino_image_tensor_list"]
            self.img_per_seq_lens = data["img_per_seq_lens"]
            self.packed_view_infos = data['packed_view_infos']
            self.packed_image_paths = data['packed_image_paths']

        if "packed_vit_images" in data.keys():
            self.packed_vit_images = data["packed_vit_images"]
            self.packed_image_grid_thw = data["packed_image_grid_thw"]
            self.packed_vit_token_indexes = data["packed_vit_token_indexes"]
            self.vit_token_seqlens = data["vit_token_seqlens"]

        if "packed_label_ids" in data.keys():
            self.packed_label_ids = data["packed_label_ids"]
            self.ce_loss_indexes = data["ce_loss_indexes"]
            self.ce_loss_weights = data["ce_loss_weights"]

    def pin_memory(self):
        self.packed_text_ids = self.packed_text_ids.pin_memory()
        self.packed_text_indexes = self.packed_text_indexes.pin_memory()
        self.packed_position_ids = self.packed_position_ids.pin_memory()

        if not self.use_flex:
            self.nested_attention_masks = [item.pin_memory() for item in self.nested_attention_masks]

        if hasattr(self, 'packed_dino_image_tensor_list'):
            self.packed_dino_token_indexes = self.packed_dino_token_indexes.pin_memory()
            self.dino_token_seqlens = self.dino_token_seqlens.pin_memory()

            self.packed_depths =  self.packed_depths.pin_memory()     
            self.packed_extrinsics =  self.packed_extrinsics.pin_memory()  
            self.packed_intrinsics =  self.packed_intrinsics.pin_memory()  
            self.packed_world_points =  self.packed_world_points.pin_memory()  
            self.packed_point_masks = self.packed_point_masks.pin_memory()  
            self.packed_dino_image_tensor_list = self.packed_dino_image_tensor_list.pin_memory()

        if hasattr(self, 'packed_vit_images'):
            self.packed_vit_images = self.packed_vit_images.pin_memory()
            self.packed_image_grid_thw = self.packed_image_grid_thw.pin_memory()
            self.packed_vit_token_indexes = self.packed_vit_token_indexes.pin_memory()
            self.vit_token_seqlens = self.vit_token_seqlens.pin_memory()

        if hasattr(self, 'packed_label_ids'):
            self.packed_label_ids = self.packed_label_ids.pin_memory()
            self.ce_loss_indexes = self.ce_loss_indexes.pin_memory()
            self.ce_loss_weights = self.ce_loss_weights.pin_memory()

        return self

    def cuda(self, device=None, non_blocking=False):
        self.packed_text_ids = self.packed_text_ids.to(device, non_blocking=non_blocking)
        self.packed_text_indexes = self.packed_text_indexes.to(device, non_blocking=non_blocking)
        self.packed_position_ids = self.packed_position_ids.to(device, non_blocking=non_blocking)

        if not self.use_flex:
            self.nested_attention_masks = [item.to(device, non_blocking=non_blocking) for item in self.nested_attention_masks]

        if hasattr(self, 'packed_dino_image_tensor_list'):
            self.packed_dino_token_indexes = self.packed_dino_token_indexes.to(device, non_blocking=non_blocking)
            self.dino_token_seqlens = self.dino_token_seqlens.to(device, non_blocking=non_blocking)

            self.packed_depths =  self.packed_depths.to(device, non_blocking=non_blocking)
            self.packed_extrinsics =  self.packed_extrinsics.to(device, non_blocking=non_blocking)
            self.packed_intrinsics =  self.packed_intrinsics.to(device, non_blocking=non_blocking)
            self.packed_world_points =  self.packed_world_points.to(device, non_blocking=non_blocking)
            self.packed_point_masks = self.packed_point_masks.to(device, non_blocking=non_blocking)
            self.packed_dino_image_tensor_list = self.packed_dino_image_tensor_list.to(device, non_blocking=non_blocking)

        if hasattr(self, 'packed_vit_images'):
            self.packed_vit_images = self.packed_vit_images.to(device, non_blocking=non_blocking)
            self.packed_image_grid_thw = self.packed_image_grid_thw.to(device, non_blocking=non_blocking)
            self.packed_vit_token_indexes = self.packed_vit_token_indexes.to(device, non_blocking=non_blocking)
            self.vit_token_seqlens = self.vit_token_seqlens.to(device, non_blocking=non_blocking)

        if hasattr(self, 'packed_label_ids'):
            self.packed_label_ids = self.packed_label_ids.to(device, non_blocking=non_blocking)
            self.ce_loss_indexes = self.ce_loss_indexes.to(device, non_blocking=non_blocking)
            self.ce_loss_weights = self.ce_loss_weights.to(device, non_blocking=non_blocking)

        return self

    def to_dict(self):
        data = dict(
            sequence_length = self.sequence_length,
            sample_lens = self.sample_lens,
            packed_text_ids = self.packed_text_ids,
            packed_text_indexes = self.packed_text_indexes,
            packed_position_ids = self.packed_position_ids,
            batch_data_indexes = self.batch_data_indexes,
        )

        if not self.use_flex:
            data['nested_attention_masks'] = self.nested_attention_masks
        else:
            data['split_lens'] = self.split_lens
            data['attn_modes'] = self.attn_modes

        if hasattr(self, 'packed_dino_image_tensor_list'):
            data['packed_dino_token_indexes'] = self.packed_dino_token_indexes
            data['dino_token_seqlens'] = self.dino_token_seqlens

            data['packed_depths'] =  self.packed_depths  
            data['packed_extrinsics'] = self.packed_extrinsics
            data['packed_intrinsics'] = self.packed_intrinsics 
            data['packed_world_points'] = self.packed_world_points 
            data['packed_point_masks'] = self.packed_point_masks
            data['packed_dino_image_tensor_list'] = self.packed_dino_image_tensor_list
            data['img_per_seq_lens'] = self.img_per_seq_lens
            data['packed_view_infos'] = self.packed_view_infos
            data['packed_image_paths'] = self.packed_image_paths

        if hasattr(self, 'packed_vit_images'):
            data['packed_vit_images'] = self.packed_vit_images
            data['packed_image_grid_thw'] = self.packed_image_grid_thw
            data['packed_vit_token_indexes'] = self.packed_vit_token_indexes
            data['vit_token_seqlens'] = self.vit_token_seqlens

        if hasattr(self, 'packed_label_ids'):
            data['packed_label_ids'] = self.packed_label_ids
            data['ce_loss_indexes'] = self.ce_loss_indexes
            data['ce_loss_weights'] = self.ce_loss_weights

        return data


def collate_wrapper():
    def collate_fn(batch):
        return SimpleCustomBatch(batch)
    return collate_fn
