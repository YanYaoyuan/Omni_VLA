import copy
from copy import deepcopy
import random
from typing import List, Optional, Tuple

from easydict import EasyDict
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.attention.flex_attention import create_block_mask
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

from ...data_vlm.data_utils import create_sparse_mask
from ...data_vlm.data_utils import get_flattened_position_ids_extrapolate
from ...data_vlm.data_utils import get_flattened_position_ids_interpolate
from ...data_vlm.data_utils import get_rope_index_image_3D
from ...data_vlm.data_utils import get_rope_index_image_3D_dino
from ...data_vlm.data_utils import patchify
from ...data_vlm.transforms_vggt import load_and_preprocess_images
from ...data_vlm.transforms_vggt import load_and_resize14
from ..pi3.models.layers.camera_head import Pi3CameraHead
from ..pi3.models.layers.pos_embed import PositionGetter
from ..pi3.models.layers.pos_embed import RoPE2D
from ..pi3.models.layers.transformer_head import Pi3ContextTransformerDecoder
from ..pi3.models.layers.transformer_head import Pi3LinearPts3d
from ..pi3.models.layers.transformer_head import Pi3TransformerDecoder
from ..pi3.models.pi3_loss import Pi3Loss
from ..pi3.utils.geometry import homogenize_points
from .modeling_utils import MLPconnector
from .modeling_utils import PositionEmbedding
from .modeling_utils import PositionEmbedding_Extra
from .modeling_utils import TimestepEmbedder
from .qwen2vl import NaiveCache

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


def freeze_all_params(modules):
    for module in modules:
        try:
            for n, param in module.named_parameters():
                param.requires_grad = False
        except AttributeError:
            # module is directly a parameter
            module.requires_grad = False


def slice_expand_and_flatten(token_tensor, B, S):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined


class G2VLMConfig(PretrainedConfig):
    def __init__(
        self,
        visual_und=True,
        visual_recon=True,
        joint_train_recon=False,
        pretrain_train_recon=False, 
        use_dinov3=False,
        ce_loss_dino=False,
        train_conf_pi3=False, 
        llm_config=None,
        vit_config=None,
        dino_config=None,
        latent_patch_size=2,
        max_latent_size=32,
        vit_max_num_patch_per_side=70,
        dino_max_num_patch_per_side=37,
        interpolate_pos=False,
        use_registers=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.visual_und = visual_und
        self.visual_recon = visual_recon
        self.train_conf_pi3 = train_conf_pi3
        self.llm_config = llm_config
        self.vit_config = vit_config
        self.dino_config = dino_config
        self.latent_patch_size = latent_patch_size
        self.max_latent_size = max_latent_size
        self.vit_max_num_patch_per_side = vit_max_num_patch_per_side
        self.dino_max_num_patch_per_side = dino_max_num_patch_per_side
        self.interpolate_pos = interpolate_pos
        self.use_registers = use_registers
        self.joint_train_recon = joint_train_recon
        self.pretrain_train_recon = pretrain_train_recon
        self.use_dinov3 = use_dinov3
        self.ce_loss_dino = ce_loss_dino


class G2VLM(PreTrainedModel):
    config_class = G2VLMConfig
    base_model_prefix = 'g2vlm'

    def __init__(self, language_model, vit_model, dino_model, config: G2VLMConfig):
        super().__init__(config)    
        self.language_model = language_model
        self.hidden_size = config.llm_config.hidden_size
        self.use_moe = "Mo" in config.llm_config.layer_module
        self.num_heads = config.llm_config.num_attention_heads

        self.conf_head = None
        self.global_point_head = None
        self.camera_head = None
        self.point_head = None
        self.use_dinov3 = config.use_dinov3
        self.ce_loss_dino = config.ce_loss_dino

        
        if config.visual_recon:
            self.dino_model = dino_model
            self.dino_patch_size = config.dino_config.patch_size #14 
            self.dino_max_num_patch_per_side = config.dino_max_num_patch_per_side
            self.dino_hidden_size = config.dino_config.hidden_size
            self.embed_dim = self.hidden_size  
            self.resnet_normalize = torchvision.transforms.Normalize(mean=_RESNET_MEAN, std=_RESNET_STD)
            self.dino2llm = nn.Linear(self.dino_hidden_size, self.hidden_size) 
            self.use_registers = config.use_registers
            self.train_conf_pi3 = config.train_conf_pi3
            if self.use_registers:
                self.register_token = nn.Parameter(torch.randn(1, 2, 4, self.hidden_size))

            if RoPE2D is None: raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
            freq = float('rope100'[len('rope'):])
            self.pi3rope = RoPE2D(freq=freq)
            self.position_getter = PositionGetter()

            if self.use_registers:
                num_register_tokens = 5
                self.patch_start_idx = num_register_tokens
                self.register_token = nn.Parameter(torch.randn(1, 1, num_register_tokens, self.hidden_size))
            else: 
                self.patch_start_idx = 0 
            self.point_decoder = Pi3TransformerDecoder(
                in_dim=self.hidden_size,   #2*self.dec_embed_dim, 
                dec_embed_dim=self.hidden_size, #1024,
                dec_num_heads=16,
                out_dim=1024,
                rope=self.pi3rope,
            )
            if self.use_dinov3:
                self.point_head = Pi3LinearPts3d(patch_size=16, dec_embed_dim=1024, output_dim=3)
            else:
                self.point_head = Pi3LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3)
            # ----------------------
            #  Camera Pose Decoder
            # ----------------------

            self.camera_decoder = Pi3TransformerDecoder(
                in_dim=self.hidden_size,
                dec_embed_dim=self.hidden_size,
                dec_num_heads=16,                
                out_dim=512,
                rope=self.pi3rope,
                use_checkpoint=False
            )
            self.camera_head = Pi3CameraHead(dim=512)
            # ----------------------
            #  Global Points Decoder
            # ----------------------
            use_global_points = True  
            self.use_global_points = use_global_points

            if use_global_points:
                self.global_points_decoder = Pi3ContextTransformerDecoder(
                    in_dim=self.hidden_size,
                    dec_embed_dim=self.hidden_size,
                    dec_num_heads=16,
                    out_dim=1024,
                    rope=self.pi3rope,
                )
                if self.use_dinov3:
                    self.global_point_head = Pi3LinearPts3d(patch_size=16, dec_embed_dim=1024, output_dim=3)
                else:
                    self.global_point_head = Pi3LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3)
            else:
                self.global_point_head = None
            
            self.Pi3Loss = Pi3Loss(self.train_conf_pi3)

            if self.train_conf_pi3:
                # assert ckpt is not None

                # ----------------------
                #     Conf Decoder
                # ----------------------
                self.conf_decoder = deepcopy(self.point_decoder)
                if self.use_dinov3:
                    self.conf_head = Pi3LinearPts3d(patch_size=16, dec_embed_dim=1024, output_dim=1)
                else:
                    self.conf_head = Pi3LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=1)

                freeze_all_params([self.dino_model, self.dino2llm, self.language_model, self.point_decoder, self.point_head, self.camera_decoder,  self.camera_head])
                freeze_all_params([self.Pi3Loss.point_loss.segformer])
                if use_global_points:
                    freeze_all_params([self.global_points_decoder, self.global_point_head])
            else:
                self.conf_head = None 

    
  
        if config.visual_und:
            self.vit_model = vit_model
            self.vit_patch_size = config.vit_config.patch_size
            self.vit_max_num_patch_per_side = 32 
            self.vit_hidden_size = config.vit_config.hidden_size
            self.use_registers = config.use_registers
       
        if config.interpolate_pos:
            self.get_flattened_position_ids = get_flattened_position_ids_interpolate
        else:
            self.get_flattened_position_ids = get_flattened_position_ids_extrapolate

        self.config = config
        self._init_weights()

    def _init_weights(self):
        if self.config.visual_recon:
            nn.init.constant_(self.dino2llm.weight, 0)
            nn.init.constant_(self.dino2llm.bias, 0)   
        if self.use_registers:
            nn.init.normal_(self.register_token, std=1e-6)

    def forward(
        self,
        sequence_length: int,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        sample_lens: List[int],
        packed_position_ids: torch.LongTensor,
        nested_attention_masks: List[torch.Tensor] = None,
        split_lens: List[int] = None,
        attn_modes: List[str] = None,
        # for visual understanding
        ce_loss_indexes: Optional[torch.BoolTensor] = None,
        packed_label_ids: Optional[torch.LongTensor] = None,
        packed_vit_tokens: Optional[torch.Tensor] = None,
        packed_vit_images: Optional[torch.Tensor] = None,
        packed_vit_token_indexes: Optional[torch.LongTensor] = None,
        packed_vit_position_ids: Optional[torch.LongTensor] = None,
        vit_token_seqlens: Optional[torch.IntTensor] = None,
        packed_image_grid_thw:  Optional[torch.IntTensor] = None,

        #reconstruct
        packed_dino_tokens: Optional[torch.Tensor] = None,
        packed_dino_token_indexes: Optional[torch.LongTensor] = None,
        packed_dino_position_ids: Optional[torch.LongTensor] = None,
        dino_token_seqlens: Optional[torch.IntTensor] = None,
        patchified_images_shapes: Optional[List[Tuple[int, int]]] = None,

        packed_dino_image_tensor_list: Optional[torch.Tensor] = None,
        packed_depths: Optional[torch.Tensor] = None,
        packed_extrinsics: Optional[torch.Tensor] = None,
        packed_intrinsics: Optional[torch.Tensor] = None,
        packed_cam_points: Optional[torch.Tensor] = None,
        packed_world_points: Optional[torch.Tensor] = None,
        packed_point_masks: Optional[torch.Tensor] = None,
        img_per_seq_lens: Optional[torch.Tensor] = None,
        query_points: Optional[torch.Tensor] = None,
        packed_view_infos=None, 
        packed_image_paths=None, 
    ) -> torch.Tensor:
        """
        Args:
            sequence_length: length of sequence.
            packed_text_ids: 1-D int tensor, packed text token ids.
            packed_text_indexes: 1-D int tensor, packed text token indexes in sequence.
            sample_lens: A list of N ints, length of each sample in packed_sequence.
            nested_attention_masks: A list of N 2-D float tensor,  where 0.0 means attention and 
                -inf means ignore.
            packed_position_ids: packed 1-D positions, an image has only one global position shared
                by all latent tokens.

            packed_vit_tokens: packed patchified image tokens for vit model.
            packed_vit_position_ids: 1-D int tensor, the position of each token for vit model.
            packed_vit_token_indexes: 1-D int tensor, packed vit token indexes in sequence.
            vit_token_seqlens: 1-D int tensor, the length of each image tokens for vit model.
            packed_label_ids: 1-D int tensor, packed label token ids.
            ce_loss_indexes: 1-D bool tensor, where to compute ce loss.

        """
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros(size=(sequence_length, self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        if self.config.visual_recon:
            sparse_mask = create_sparse_mask(sample_lens, split_lens, attn_modes, packed_text_embedding.device)
            seqlen = sum(sample_lens)
            block_mask = create_block_mask(
                sparse_mask, B=1, H=self.num_heads, Q_LEN=seqlen, KV_LEN=seqlen, 
                device=packed_text_embedding.device, BLOCK_SIZE=128, _compile=True
            )
            attention_mask = block_mask
        else:### regular mask 
            if nested_attention_masks is None:
                sparse_mask = create_sparse_mask(sample_lens, split_lens, attn_modes, packed_text_embedding.device)
                seqlen = sum(sample_lens)
                block_mask = create_block_mask(
                    sparse_mask, B=1, H=self.num_heads, Q_LEN=seqlen, KV_LEN=seqlen, 
                    device=packed_text_embedding.device, BLOCK_SIZE=128, _compile=True
                )
                attention_mask = block_mask
            else:
                attention_mask = nested_attention_masks

        if self.config.visual_und:
            cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0))
            cu_seqlens = cu_seqlens.to(torch.int32)
            max_seqlen = torch.max(vit_token_seqlens).item()


            image_embeds = self.vit_model(packed_vit_images, grid_thw=packed_image_grid_thw)

            packed_vit_token_embed = image_embeds

            packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed  

        if self.config.visual_recon and dino_token_seqlens is not None:
 
            cu_seqlens = torch.nn.functional.pad(torch.cumsum(dino_token_seqlens, dim=0), (1, 0))
            cu_seqlens = cu_seqlens.to(torch.int32)
            max_seqlen = torch.max(dino_token_seqlens).item()
      

            BS, C_in, H, W = packed_dino_image_tensor_list.shape
        
            S = img_per_seq_lens[0] #constant for now 
            B = BS // S 
            if self.config.joint_train_recon or self.config.pretrain_train_recon:
                images = packed_dino_image_tensor_list.reshape(B, S, C_in, H, W)
                if self.use_dinov3:
                    patch_h, patch_w = H // 16, W // 16
                else:
                    patch_h, patch_w = H // 14, W // 14

                ## undo resnet_norm 
                resmean = torch.tensor(_RESNET_MEAN).view(1, 1, -1, 1, 1).to(images.device)
                resstd = torch.tensor(_RESNET_STD).view(1, 1, -1, 1, 1).to(images.device)
                images_unorm = images * resstd + resmean 
        
                batch = {}
                batch['depths'] = packed_depths.reshape(B, S, *packed_depths.shape[1:]).to(torch.float32)
                batch['extrinsics'] = packed_extrinsics.reshape(B, S, *packed_extrinsics.shape[1:]).to(torch.float32)
                batch['intrinsics'] = packed_intrinsics.reshape(B, S, *packed_intrinsics.shape[1:]).to(torch.float32)
                # batch['cam_points'] = packed_cam_points.reshape(B, S, *packed_cam_points.shape[1:])
                batch['world_points'] = packed_world_points.reshape(B, S, *packed_world_points.shape[1:]).to(torch.float32)
                batch['point_masks'] = packed_point_masks.reshape(B, S, *packed_point_masks.shape[1:])
                batch['images'] = images_unorm
                batch['view_infos'] = packed_view_infos
                batch['image_paths'] = packed_image_paths
     
            if self.use_dinov3:
                packed_dino_token_embed = self.dino_model(
                    # packed_pixel_values=packed_dino_image_tensor_list, 
                    pixel_values=packed_dino_image_tensor_list,
                    # packed_flattened_position_ids=packed_vit_position_ids,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                )
            else:
                packed_dino_token_embed = self.dino_model(
                    packed_pixel_values=packed_dino_image_tensor_list, 
                    # packed_flattened_position_ids=packed_vit_position_ids,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                )


            BS, P, D = packed_dino_token_embed.size() #
            packed_dino_token_embed = packed_dino_token_embed.reshape(BS*P, D)
            packed_dino_token_embed = self.dino2llm(packed_dino_token_embed)


            _, D = packed_dino_token_embed.shape
            packed_dino_token_embed = packed_dino_token_embed.reshape(BS, -1, D) 
         
            if self.use_registers:
                register_token = self.register_token.repeat(B, S, 1, 1).reshape(B*S, *self.register_token.shape[-2:])
                packed_dino_token_embed = torch.cat([register_token, packed_dino_token_embed], dim=1)

            packed_dino_token_embed = packed_dino_token_embed.reshape(-1, D)

            packed_sequence[packed_dino_token_indexes] = packed_dino_token_embed
            
        extra_inputs = {}
        if self.use_moe:
            packed_und_token_indexes = packed_text_indexes
            if packed_vit_token_indexes is not None:
                packed_und_token_indexes=torch.cat([packed_text_indexes, packed_vit_token_indexes], dim=0)
            extra_inputs.update(
                packed_und_token_indexes=packed_und_token_indexes,
                packed_geo_token_indexes=packed_dino_token_indexes, 
            )

        if self.config.visual_recon and dino_token_seqlens is not None:

            last_hidden_state = self.language_model(
                packed_sequence=packed_sequence,
                sample_lens=sample_lens,
                attention_mask=attention_mask,
                packed_position_ids=packed_position_ids,
                output_hidden_states=True,
                **extra_inputs,
            )
            selected_hidden_states = last_hidden_state.hidden_states
            last_hidden_state = last_hidden_state.packed_query_sequence
        else: 
            last_hidden_state = self.language_model(
                packed_sequence=packed_sequence,
                sample_lens=sample_lens,
                attention_mask=attention_mask,
                packed_position_ids=packed_position_ids,
                output_hidden_states=False,
                **extra_inputs,
            )
            last_hidden_state = last_hidden_state.packed_query_sequence


        vggt_loss_dict = None
        dl_loss = 0 
        details = {}
        predictions = {}
  
        if self.config.visual_recon:
 
            if self.config.joint_train_recon or self.config.pretrain_train_recon:
                predictions['world_points'] = batch['world_points']
                predictions['point_masks'] = batch['point_masks']
                predictions['view_infos'] = batch['view_infos']
                predictions['image_paths'] = batch['image_paths']
      
                N = S 
                hidden = last_hidden_state[packed_dino_token_indexes].reshape(B*S, -1, D)
                hw = hidden.shape[1]
     
                if self.use_dinov3:
                    pos = self.position_getter(B * N, H//16, W//16, hidden.device)
                else:
                    pos = self.position_getter(B * N, H//14, W//14, hidden.device)
                if self.patch_start_idx > 0:
                  
                    pos = pos + 1
                    pos_special = torch.zeros(B * N, self.patch_start_idx, 2).to(hidden.device).to(pos.dtype)
                    pos = torch.cat([pos_special, pos], dim=1)
            
                pos = pos.reshape(B*N, hw, -1)

                point_hidden = self.point_decoder(hidden, xpos=pos)
                if self.train_conf_pi3:
                    conf_hidden = self.conf_decoder(hidden, xpos=pos)
                camera_hidden = self.camera_decoder(hidden, xpos=pos)
                if self.use_global_points:
                    context = hidden.reshape(B, N, patch_h*patch_w+self.patch_start_idx, -1)[:, 0:1].repeat(1, N, 1, 1).reshape(B*N, patch_h*patch_w+self.patch_start_idx, -1)
                    global_point_hidden = self.global_points_decoder(hidden, context, xpos=pos, ypos=pos)
                
                with torch.amp.autocast(device_type='cuda', enabled=False):
                    # local points
                    point_hidden = point_hidden.float()
                    ret = self.point_head([point_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
                    xy, z = ret.split([2, 1], dim=-1)
                    z = torch.exp(z)
                    local_points = torch.cat([xy * z, z], dim=-1)

                    # confidence
                    if self.train_conf_pi3:
                        conf_hidden = conf_hidden.float()
                        conf = self.conf_head([conf_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
                    else:
                        conf = None
                        
                    # camera
                    camera_hidden = camera_hidden.float()
                    camera_poses = self.camera_head(camera_hidden[:, self.patch_start_idx:], patch_h, patch_w).reshape(B, N, 4, 4)

                    # Global points
                    if self.use_global_points:
                        global_point_hidden = global_point_hidden.float()
                        global_points = self.global_point_head([global_point_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
                    else:
                        global_points = None
                    # unproject local points using camera poses
                    points = torch.einsum('bnij, bnhwj -> bnhwi', camera_poses, homogenize_points(local_points))[..., :3]
                
                pi3_pred = dict(
                    points=points,
                    local_points=local_points,
                    conf=conf,
                    camera_poses=camera_poses,
                    global_points=global_points
                )
                predictions['points'] = points
                predictions['camera_poses'] = camera_poses
                predictions['local_points'] = local_points
                predictions['global_points'] = global_points

                with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16): 
                    dl_loss, details = self.Pi3Loss(pi3_pred, batch)


                predictions["images"] = images_unorm

        ce = None
        if ce_loss_indexes is not None:
    
            packed_ce_preds = self.language_model.lm_head(last_hidden_state[ce_loss_indexes])

            ce = F.cross_entropy(packed_ce_preds, packed_label_ids, reduction="none") #note because here it employed 

        
        if vggt_loss_dict is not None:
            if 'loss_reg_point' in vggt_loss_dict: 
                return dict(mse=mse, ce=ce, dl=vggt_loss, depth_loss_reg=vggt_depth_loss_reg, \
                            depth_loss_conf=vggt_depth_loss_conf, point_loss_reg=vggt_point_loss_reg, \
                            camera_loss=vggt_camera_loss, point_loss_conf=vggt_point_loss_conf, \
                            camera_auc_30=camera_auc_30, camera_auc_20=camera_auc_20, camera_auc_10=camera_auc_10,\
                                camera_auc_5=camera_auc_5,camera_auc_3=camera_auc_3), predictions
            else: 
                return dict(mse=mse, ce=ce, dl=vggt_loss, depth_loss_reg=vggt_depth_loss_reg, \
                            depth_loss_conf=vggt_depth_loss_conf, \
                            camera_loss=vggt_camera_loss, \
                            camera_auc_30=camera_auc_30, camera_auc_20=camera_auc_20, camera_auc_10=camera_auc_10,\
                                camera_auc_5=camera_auc_5,camera_auc_3=camera_auc_3), predictions
        else: 
    
            return  EasyDict(
                mse=mse, 
                ce=ce,
                dl=dl_loss,
                **details
            ), predictions 


    def prepare_prompts_addbos(self, curr_kvlens, curr_rope, prompts, tokenizer, new_token_ids):
        packed_text_ids = list()
        packed_text_position_ids = list()
        text_token_lens = list()
        packed_text_indexes = list()
        packed_key_value_indexes = list()

        curr = 0
        newlens, new_rope = list(), list()
        for prompt, curr_kvlen, curr_position_id in zip(prompts, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            text_ids = tokenizer.encode(prompt)
            text_ids = [new_token_ids['bos_token_id']] + text_ids 
            text_token_lens.append(len(text_ids))
            packed_text_ids.extend(text_ids)
            packed_text_position_ids.extend(range(curr_position_id, curr_position_id + len(text_ids)))
            packed_text_indexes.extend(range(curr, curr + len(text_ids)))
            newlens.append(curr_kvlen + len(text_ids))
            new_rope.append(curr_position_id + len(text_ids))
            curr += len(text_ids)
        

        generation_input = {
            "text_token_lens": torch.tensor(text_token_lens, dtype=torch.int),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_position_ids": torch.tensor(packed_text_position_ids, dtype=torch.long).expand(3, -1),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    def prepare_prompts_addeos(self, curr_kvlens, curr_rope, prompts, tokenizer, new_token_ids):
        packed_text_ids = list()
        packed_text_position_ids = list()
        text_token_lens = list()
        packed_text_indexes = list()
        packed_key_value_indexes = list()

        curr = 0
        newlens, new_rope = list(), list()
        for prompt, curr_kvlen, curr_position_id in zip(prompts, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            text_ids = tokenizer.encode(prompt)
            assistant_ids = tokenizer.encode('assistant\n')
            text_ids = text_ids + [new_token_ids['eos_token_id']] + [new_token_ids['bos_token_id']] + assistant_ids
            text_token_lens.append(len(text_ids))
            packed_text_ids.extend(text_ids)
            packed_text_position_ids.extend(range(curr_position_id, curr_position_id + len(text_ids)))
            packed_text_indexes.extend(range(curr, curr + len(text_ids)))
            newlens.append(curr_kvlen + len(text_ids))
            new_rope.append(curr_position_id + len(text_ids))
            curr += len(text_ids)
        

        generation_input = {
            "text_token_lens": torch.tensor(text_token_lens, dtype=torch.int),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_position_ids": torch.tensor(packed_text_position_ids, dtype=torch.long).expand(3, -1),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope
    
    def prepare_prompts_pure_text(self, curr_kvlens, curr_rope, prompts, tokenizer, new_token_ids):
        packed_text_ids = list()
        packed_text_position_ids = list()
        text_token_lens = list()
        packed_text_indexes = list()
        packed_key_value_indexes = list()

        curr = 0
        newlens, new_rope = list(), list()
        for prompt, curr_kvlen, curr_position_id in zip(prompts, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            text_ids = tokenizer.encode(prompt)
            text_token_lens.append(len(text_ids))
            packed_text_ids.extend(text_ids)
            packed_text_position_ids.extend(range(curr_position_id, curr_position_id + len(text_ids)))
            packed_text_indexes.extend(range(curr, curr + len(text_ids)))
            newlens.append(curr_kvlen + len(text_ids))
            new_rope.append(curr_position_id + len(text_ids))
            curr += len(text_ids)
        

        generation_input = {
            "text_token_lens": torch.tensor(text_token_lens, dtype=torch.int),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_position_ids": torch.tensor(packed_text_position_ids, dtype=torch.long).expand(3, -1),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope
    
    def prepare_prompts(self, curr_kvlens, curr_rope, prompts, tokenizer, new_token_ids):
        packed_text_ids = list()
        packed_text_position_ids = list()
        text_token_lens = list()
        packed_text_indexes = list()
        packed_key_value_indexes = list()

        curr = 0
        newlens, new_rope = list(), list()
        for prompt, curr_kvlen, curr_position_id in zip(prompts, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            text_ids = tokenizer.encode(prompt)
            text_ids = [new_token_ids['bos_token_id']] + text_ids + [new_token_ids['eos_token_id']]
            text_token_lens.append(len(text_ids))
            packed_text_ids.extend(text_ids)
            packed_text_position_ids.extend(range(curr_position_id, curr_position_id + len(text_ids)))
            packed_text_indexes.extend(range(curr, curr + len(text_ids)))
            newlens.append(curr_kvlen + len(text_ids))
            new_rope.append(curr_position_id + len(text_ids))
            curr += len(text_ids)
        

        generation_input = {
            "text_token_lens": torch.tensor(text_token_lens, dtype=torch.int),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_position_ids": torch.tensor(packed_text_position_ids, dtype=torch.long).expand(3, -1),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    @torch.no_grad
    def forward_cache_update_text(
        self,
        past_key_values: NaiveCache,
        packed_text_ids: torch.IntTensor,
        packed_text_position_ids: torch.LongTensor,
        text_token_lens: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
    ):  
 
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {"mode": "und"}

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_text_embedding,
            query_lens=text_token_lens,
            packed_query_position_ids=packed_text_position_ids,
            packed_query_indexes=packed_text_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            update_past_key_values=True,
            is_causal=True,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values

    def prepare_vit_images(self, curr_kvlens, curr_rope, images, transforms, new_token_ids):
        packed_vit_token_indexes = list()
        vit_token_seqlens, packed_vit_tokens, packed_vit_position_ids = list(), list(), list()
        packed_vit_images = list()
        packed_image_grid_thw = list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        _curr = curr = 0
        newlens, new_rope = list(), list()
        for image, curr_kvlen, curr_position_id in zip(images, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1
         
            pos_tensor = torch.full((1,), curr_position_id, dtype=torch.long)
            packed_position_ids.extend([pos_tensor.expand(3, 1)])
            curr_position_id += 1

            image_tensor,  image_grid_thw = transforms([image])
            packed_image_grid_thw.append(image_grid_thw[0])
            num_img_tokens = image_tensor.shape[0] // 4 
            packed_vit_images.append(image_tensor)
            
            vit_token_seqlens.append(num_img_tokens)
            packed_vit_token_indexes.extend(range(_curr, _curr + num_img_tokens))
            packed_indexes.extend(range(curr, curr + num_img_tokens))
            curr += num_img_tokens
            _curr += num_img_tokens


            postions_ids_from_vit_for_rope, rope_deltas = get_rope_index_image_3D(
                image_grid_thw[0],
                curr_position_id,
                device=image_tensor.device
            )

            packed_position_ids.extend([postions_ids_from_vit_for_rope])
            curr_position_id += rope_deltas + 1

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            pos_tensor = torch.full((1,), curr_position_id, dtype=torch.long)
            packed_position_ids.extend([pos_tensor.expand(3, 1)])
            curr_position_id += 1


            packed_seqlens.append(num_img_tokens + 2)
            newlens.append(curr_kvlen + num_img_tokens + 2)
            new_rope.append(curr_position_id)

        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "vit_token_seqlens": torch.tensor(vit_token_seqlens, dtype=torch.int),
            "packed_image_grid_thw": torch.stack(packed_image_grid_thw, dim=0),
            "packed_vit_images": torch.stack(packed_vit_images, dim=0),
            "packed_vit_token_indexes": torch.tensor(packed_vit_token_indexes, dtype=torch.long),
            "packed_position_ids": torch.cat(packed_position_ids, dim=1),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    @torch.no_grad
    def forward_cache_update_vit(
        self,
        past_key_values: NaiveCache,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_vit_images: torch.Tensor,
        packed_image_grid_thw:  torch.IntTensor,
        packed_vit_token_indexes: torch.LongTensor,
        vit_token_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_vit_tokens: Optional[torch.Tensor]=None,
        packed_vit_position_ids: Optional[torch.LongTensor] = None,
    ):  

        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0))
        cu_seqlens = cu_seqlens.to(torch.int32)
        max_seqlen = torch.max(vit_token_seqlens).item()

        image_embeds = self.vit_model(packed_vit_images, grid_thw=packed_image_grid_thw)
        packed_vit_token_embed = image_embeds



        if packed_vit_token_embed.dtype != packed_sequence.dtype:
            packed_vit_token_embed = packed_vit_token_embed.to(packed_sequence.dtype)
        packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {"mode": "und"}

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            update_past_key_values=True,
            is_causal=False,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values
    
    def prepare_dino_images_pi3 (self, curr_kvlens, curr_rope, images, transforms, new_token_ids):
        packed_dino_token_indexes = list()
        dino_token_seqlens, packed_dino_tokens, packed_dino_position_ids = list(), list(), list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        _curr = curr = 0
        newlens, new_rope = list(), list()
    
        vggt_fixed_resolution = 518 # hardcode 
        img_load_resolution = 1024

        images = load_and_resize14(images,vggt_fixed_resolution)

        curr_kvlen = curr_kvlens[0]
        curr_position_id = curr_rope[0]
        
        packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
        curr += curr_kvlen
        for image in images:
            
            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            image_tensor = image
            height, width = image_tensor.shape[1:]
            grid_t = 1  
            grid_h, grid_w = height // 14, width // 14

            # add 3d pos for <|startofimage|> token
            pos_tensor = torch.full((1,), curr_position_id, dtype=torch.long)
            packed_position_ids.extend([pos_tensor.expand(3, 1)])
            curr_position_id += 1

            dino_tokens = patchify(image_tensor, self.dino_patch_size)
            packed_dino_tokens.append(dino_tokens)
            num_img_tokens = dino_tokens.shape[0]
            dino_token_seqlens.append(num_img_tokens)
  
            packed_dino_token_indexes.extend(range(_curr, _curr + num_img_tokens))
            packed_indexes.extend(range(curr, curr + num_img_tokens))
            curr += num_img_tokens
            _curr += num_img_tokens

            ###3d rope embedding for QKV attention: 
            dino_image_thw = torch.tensor([grid_t, grid_h, grid_w], dtype=torch.long) 
            postions_ids_from_dino_for_rope, rope_deltas = get_rope_index_image_3D_dino(
                dino_image_thw,
                curr_position_id,
                device=image_tensor.device
            )
            packed_position_ids.extend([postions_ids_from_dino_for_rope])
            curr_position_id += rope_deltas + 1

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            pos_tensor = torch.full((1,), curr_position_id, dtype=torch.long)
            packed_position_ids.extend([pos_tensor.expand(3, 1)])
            curr_position_id += 1

            packed_seqlens.append(num_img_tokens + 2)
            newlens.append(curr_kvlen + num_img_tokens + 2)
            curr_kvlen += num_img_tokens + 2

            new_rope.append(curr_position_id)

        newlens = [newlens[-1]]
        new_rope = [new_rope[-1]]
        packed_seqlens = [sum(packed_seqlens)]


        assert len(images.shape) == 4
        assert images.shape[1] == 3
        original_images = images.clone()
        images = torchvision.transforms.Normalize(mean=_RESNET_MEAN, std=_RESNET_STD)(images) 

        generation_input = {
            "packed_dino_images": images, 
            'original_images': original_images,
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "dino_token_seqlens": torch.tensor(dino_token_seqlens, dtype=torch.int),
            "packed_dino_token_indexes": torch.tensor(packed_dino_token_indexes, dtype=torch.long),
            "packed_position_ids": torch.cat(packed_position_ids, dim=1),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }
    
        return generation_input, newlens, new_rope

    @torch.no_grad
    def forward_cache_update_dino(
        self,
        past_key_values: NaiveCache,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_dino_token_indexes: torch.LongTensor,
        dino_token_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_dino_images: torch.Tensor, 
        original_images: torch.Tensor, 
    ):
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding
        
        cu_seqlens = torch.nn.functional.pad(torch.cumsum(dino_token_seqlens, dim=0), (1, 0))
        cu_seqlens = cu_seqlens.to(torch.int32)
        max_seqlen = torch.max(dino_token_seqlens).item()
 
        packed_dino_token_embed = self.dino_model(
            packed_pixel_values=packed_dino_images, 
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        B, P, D = packed_dino_token_embed.size() #
        packed_dino_token_embed = packed_dino_token_embed.reshape(B*P, D)

        packed_dino_token_embed = self.dino2llm(packed_dino_token_embed)

        BS, C_in, H, W = packed_dino_images.shape
        S = BS ### constant for now 
        B = BS // S 
        assert B==1

        if packed_dino_token_embed.dtype != packed_sequence.dtype:
            packed_dino_token_embed = packed_dino_token_embed.to(packed_sequence.dtype)
        packed_sequence[packed_dino_token_indexes] = packed_dino_token_embed

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {
                "mode": "geo",
                "packed_geo_token_indexes": packed_dino_token_indexes, 
                "packed_text_indexes": packed_text_indexes
            } 

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            update_past_key_values=True,
            output_hidden_states=False, 
            is_causal=False,
            **extra_inputs,
        )

     
        past_key_values = output.past_key_values
        last_hidden_state = output.packed_query_sequence


        return past_key_values, last_hidden_state


    def prepare_start_tokens(self, curr_kvlens, curr_rope, tokenizer, new_token_ids):
        packed_start_tokens, packed_key_value_indexes = list(), list()
        packed_query_position_ids = list()

        template = "<|im_start|>user\your text<|im_end|>\n<|im_start|>assistant\n"
        
        template_ids = tokenizer.encode(template, add_special_tokens=False)
        if template_ids:
            start_token_id = template_ids[-1]  
        else:
            start_token_id = tokenizer.eos_token_id or 151643 

        curr = 0
        for curr_kvlen, curr_position_id in zip(curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            packed_start_tokens.append(start_token_id)
            packed_query_position_ids.append(curr_position_id)
            curr += curr_kvlen

        generation_input = {
            "packed_start_tokens": torch.tensor(packed_start_tokens, dtype=torch.long),
            "packed_query_position_ids": torch.tensor(packed_query_position_ids, dtype=torch.long).expand(3, -1),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
        }

        return generation_input

    @torch.no_grad
    def generate_text(
        self,
        past_key_values: NaiveCache,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_start_tokens: torch.LongTensor,
        packed_query_position_ids: torch.LongTensor,
        max_length: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        end_token_id: int = None,
    ):
        step = 0
        generated_sequence = []
        curr_tokens = packed_start_tokens
        while step < max_length:
            generated_sequence.append(curr_tokens)
            packed_text_embedding = self.language_model.model.embed_tokens(curr_tokens)
            query_lens = torch.ones_like(curr_tokens)
            packed_query_indexes = torch.cumsum(key_values_lens, dim=0) + torch.arange(
                0, len(key_values_lens), 
                device=key_values_lens.device, 
                dtype=key_values_lens.dtype
            )

            uppacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
            for i in range(len(uppacked)):
                uppacked[i] += i
            packed_key_value_indexes = torch.cat(uppacked, dim=0)

            extra_inputs = {}
            if self.use_moe:
                extra_inputs = {"mode": "und"}

            output = self.language_model.forward_inference(
                packed_query_sequence=packed_text_embedding,
                query_lens=query_lens,
                packed_query_position_ids=packed_query_position_ids,
                packed_query_indexes=packed_query_indexes,
                past_key_values=past_key_values,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                update_past_key_values=True,
                is_causal=True,
                **extra_inputs,
            )
            past_key_values = output.past_key_values
            packed_query_sequence = output.packed_query_sequence
            pred_logits = self.language_model.lm_head(packed_query_sequence)

            if do_sample:
                probs = nn.functional.softmax(pred_logits / temperature, dim=-1)
                curr_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                curr_tokens = torch.argmax(pred_logits, dim=-1)

            uppacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
            for i in range(len(uppacked)):
                uppacked[i] = torch.cat(
                    [uppacked[i], torch.tensor([uppacked[i][-1] + 1], device=uppacked[i].device)], dim=0
                )
            packed_key_value_indexes = torch.cat(uppacked, dim=0)
            key_values_lens = key_values_lens + 1
            packed_query_position_ids = packed_query_position_ids + 1
            step += 1

            if end_token_id is not None and curr_tokens[0] == end_token_id: # only support batch=1
                break

        output_device = generated_sequence[0].device
        return torch.stack([i.to(output_device) for i in generated_sequence], dim=0)

    @torch.no_grad
    def reconstruct(
        self,
        past_key_values: NaiveCache,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        selected_hidden_states: torch.Tensor, 
        packed_dino_token_indexes: torch.Tensor,
        packed_dino_images: torch.Tensor,
        original_images: torch.Tensor,
        **kwargs,
    ):
        ps_idx = 0  # hardcode 
        BS, C_in, H, W = packed_dino_images.shape
        B = 1
        S = BS // B
        N = S
        if len(original_images.shape) == 4:
            original_images = original_images.unsqueeze(0)
        dino_images=packed_dino_images[None]
        print('original_images', original_images.shape)
        print('dino_images', dino_images.shape)

        aggregated_tokens_list = []
        _, D = selected_hidden_states.shape ### this is actually only last hidden 
        hidden = selected_hidden_states[packed_dino_token_indexes].reshape(B*S, -1, D)

    
        hw = hidden.shape[1]
        if self.use_dinov3:
            pos = self.position_getter(B * N, H//16, W//16, hidden.device)
            patch_h, patch_w = H // 16, W // 16
        else:
            pos = self.position_getter(B * N, H//14, W//14, hidden.device)
            patch_h, patch_w = H // 14, W // 14

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * N, self.patch_start_idx, 2).to(hidden.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)
        
        pos = pos.reshape(B*N, hw, -1)

        ### return original images
        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            point_hidden = self.point_decoder(hidden, xpos=pos)
            if self.conf_head is not None:
                conf_hidden = self.conf_decoder(hidden, xpos=pos)
            camera_hidden = self.camera_decoder(hidden, xpos=pos)
            if self.use_global_points:
                context = hidden.reshape(B, N, patch_h*patch_w+self.patch_start_idx, -1)[:, 0:1].repeat(1, N, 1, 1).reshape(B*N, patch_h*patch_w+self.patch_start_idx, -1)
                global_point_hidden = self.global_points_decoder(hidden, context, xpos=pos, ypos=pos)
            
            # local points
            with torch.amp.autocast(device_type='cuda', enabled=False):
                point_hidden = point_hidden.float()
                ret = self.point_head([point_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
                xy, z = ret.split([2, 1], dim=-1)
                z = torch.exp(z)
                local_points = torch.cat([xy * z, z], dim=-1)

                # confidence
                if self.conf_head is not None:
                    conf_hidden = conf_hidden.float()
                    conf = self.conf_head([conf_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
                else:
                    conf = None
                    
                # camera
                camera_hidden = camera_hidden.float()
                camera_poses = self.camera_head(camera_hidden[:, self.patch_start_idx:], patch_h, patch_w).reshape(B, N, 4, 4)

                # Global points
                if self.use_global_points:
                    global_point_hidden = global_point_hidden.float()
                    global_points = self.global_point_head([global_point_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
                else:
                    global_points = None
                
                # unproject local points using camera poses
                points = torch.einsum('bnij, bnhwj -> bnhwi', camera_poses, homogenize_points(local_points))[..., :3]

        pi3_pred = dict(
            points=points,
            local_points=local_points,
            conf=conf,
            camera_poses=camera_poses,
            global_points=global_points
            
        )
        pi3_pred['images'] = original_images
    
        return pi3_pred
  
    @torch.no_grad()
    def recon(
        self,
        tokenizer,
        new_token_ids,
        dino_image_transform,
        images, #this now expect image paths 
        prompt='Reconstruct the 3D scene.',
    ):
        device = next(self.parameters()).device

        if isinstance(new_token_ids, dict):
            for k, v in new_token_ids.items():
                if torch.is_tensor(v):
                    new_token_ids[k] = v.to(device)
        elif torch.is_tensor(new_token_ids):
            new_token_ids = new_token_ids.to(device)

        # prefill
        past_key_values = NaiveCache(self.config.llm_config.num_hidden_layers)
        newlens = [0]
        new_rope = [0]

        # system_prompt = 'system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>assistant\n'
        system_prompt = 'Reconstruct the 3D scene.'

        print('Prepareing prompt')
        generation_input, newlens, new_rope = self.prepare_prompts_addbos(
            curr_kvlens=newlens,
            curr_rope=new_rope, 
            prompts=[system_prompt],
            tokenizer=tokenizer, 
            new_token_ids=new_token_ids,
        )
        for k, v in generation_input.items():
            if torch.is_tensor(v):
                generation_input[k] = v.to(device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values = self.forward_cache_update_text(past_key_values, **generation_input)

        print('Prepareing dino images ')

        # generation_input, newlens, new_rope = self.prepare_dino_images_none(
        generation_input, newlens, new_rope = self.prepare_dino_images_pi3(
            curr_kvlens=newlens,
            curr_rope=new_rope, 
            images=images, 
            transforms=dino_image_transform,
            new_token_ids=new_token_ids,
        )
        for k, v in generation_input.items():
            if torch.is_tensor(v):
                generation_input[k] = v.to(device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values, last_hidden_state = self.forward_cache_update_dino(past_key_values, **generation_input)

        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            predictions_dict = self.reconstruct(
                past_key_values=past_key_values,
                selected_hidden_states=last_hidden_state,
                **generation_input,
            )

        return predictions_dict

    @torch.no_grad()
    def chat_with_recon(
        self,
        tokenizer,
        new_token_ids,
        image_transform,
        dino_image_transform,
        images,
        prompt,
        max_length: int,
        do_sample: bool = False,
        temperature: float = 1.0,
    ):
        device = next(self.parameters()).device

        if isinstance(new_token_ids, dict):
            for k, v in new_token_ids.items():
                if torch.is_tensor(v):
                    new_token_ids[k] = v.to(device)
        elif torch.is_tensor(new_token_ids):
            new_token_ids = new_token_ids.to(device)

        # prefill
        past_key_values = NaiveCache(self.config.llm_config.num_hidden_layers)
        newlens = [0]
        new_rope = [0]

        #add text system prompt hard code: 
        system_prompt = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'
        generation_input, newlens, new_rope = self.prepare_prompts_pure_text(
            curr_kvlens=newlens,
            curr_rope=new_rope, 
            prompts=[system_prompt],
            tokenizer=tokenizer, 
            new_token_ids=new_token_ids,
        )
        for k, v in generation_input.items():
            if torch.is_tensor(v):
                generation_input[k] = v.to(device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values = self.forward_cache_update_text(past_key_values, **generation_input)
            
        generation_input, newlens, new_rope = self.prepare_dino_images_pi3(
            curr_kvlens=newlens,
            curr_rope=new_rope, 
            images=images.copy(), 
            transforms=dino_image_transform,
            new_token_ids=new_token_ids,
        )
        tmp_save = {}
        for k, v in generation_input.items():
            if torch.is_tensor(v):
                generation_input[k] = v.to(device)
                tmp_save[k] = v.to(device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values, last_hidden_state = self.forward_cache_update_dino(past_key_values, **generation_input)
            
        for image in images:

            generation_input, newlens, new_rope = self.prepare_vit_images(
                curr_kvlens=newlens,
                curr_rope=new_rope, 
                images=[image], 
                transforms=image_transform,
                new_token_ids=new_token_ids,
            )
            for k, v in generation_input.items():
                if torch.is_tensor(v):
                    generation_input[k] = v.to(device)
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                past_key_values = self.forward_cache_update_vit(past_key_values, **generation_input)

        # add text  
        prompt = prompt+'<|im_end|>\n<|im_start|>assistant'
        generation_input, newlens, new_rope = self.prepare_prompts_pure_text( #self.prepare_prompts_addeos(
            curr_kvlens=newlens,
            curr_rope=new_rope, 
            prompts=[prompt],
            tokenizer=tokenizer, 
            new_token_ids=new_token_ids,
        )
        for k, v in generation_input.items():
            if torch.is_tensor(v):
                generation_input[k] = v.to(device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values = self.forward_cache_update_text(past_key_values, **generation_input)


        # decode
        generation_input = self.prepare_start_tokens(newlens, new_rope,tokenizer, new_token_ids)
        for k, v in generation_input.items():
            if torch.is_tensor(v):
                generation_input[k] = v.to(device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            unpacked_latent = self.generate_text(
                past_key_values=past_key_values,
                max_length=max_length,
                do_sample=do_sample,
                temperature=temperature,
                end_token_id=new_token_ids['eos_token_id'],
                **generation_input,
            )
        
        # skip the start token
        output = tokenizer.decode(unpacked_latent[1:,0])
        return output

    def get_image_features(self, pixel_values: torch.Tensor):
        """_summary_
            Get Image last hidden states from ViT & Dino model and project to LLM hidden size.
        Args:
            pixel_values (torch.Tensor): _description_
        Return:
            image_features (torch.Tensor): _description_ Image featyre tensor of shape '(num_images, image_length, embed_dim)'
        """
        device = pixel_values.device
        