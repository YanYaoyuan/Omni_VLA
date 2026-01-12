import math
import random
from PIL import Image

import torch
from torch.nn.attention.flex_attention import or_masks, and_masks
import re


def create_sparse_mask(document_lens, split_lens, attn_modes, device):
    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    def full_and_noise_mask(b, h, q_idx, kv_idx):
        return (full_and_noise_seq_id[q_idx] == full_and_noise_seq_id[kv_idx]) & (full_and_noise_seq_id[q_idx] >= 0)

    def remove_noise_mask(b, h, q_idx, kv_idx):
        return (~((noise_seq_id[kv_idx] >= 0) & (noise_seq_id[q_idx] != noise_seq_id[kv_idx])))
 
    def sample_mask(b, h, q_idx, kv_idx):
        return document_id[q_idx] == document_id[kv_idx]

    full_and_noise_tmp = []
    noise_tmp = []

    for i, (length, model) in enumerate(zip(split_lens, attn_modes)):
        value = i if model in ['full', 'noise'] else -1
        full_and_noise_tmp.extend([value] * length)
        value_noise = i if model == 'noise' else -1
        noise_tmp.extend([value_noise] * length)

    full_and_noise_seq_id = torch.Tensor(full_and_noise_tmp).to(device)
    noise_seq_id = torch.Tensor(noise_tmp).to(device)

    document_id = torch.cat([torch.full((l,), i) for i, l in enumerate(document_lens, start=1)]).to(device)

    return and_masks(or_masks(causal_mask, full_and_noise_mask), remove_noise_mask, sample_mask)


def patchify(image, patch_size):
    p = patch_size
    c, h, w = image.shape
    assert h % p == 0 and w % p == 0
    image = image.reshape(c, h // p, p, w // p, p)
    image = torch.einsum("chpwq->hwpqc", image)
    image = image.reshape(-1, p**2 * c)
    return image


def get_flattened_position_ids_extrapolate(img_h, img_w, patch_size, max_num_patches_per_side):
    num_patches_h, num_patches_w = img_h // patch_size, img_w // patch_size
    coords_h = torch.arange(0, num_patches_h)
    coords_w = torch.arange(0, num_patches_w)
    pos_ids = (coords_h[:, None] * max_num_patches_per_side + coords_w).flatten()
    return pos_ids

# if (height, width) not in self.position_cache:
#     y_coords = torch.arange(height, device=device)
#     x_coords = torch.arange(width, device=device)
#     positions = torch.cartesian_prod(y_coords, x_coords)
#     self.position_cache[height, width] = positions

# cached_positions = self.position_cache[height, width]
# return cached_positions.view(1, height * width, 2).expand(batch_size, -1, -1).clone()

def get_flattened_position_ids_interpolate(img_h, img_w, patch_size, max_num_patches_per_side):
    num_patches_h, num_patches_w = img_h // patch_size, img_w // patch_size
    boundaries = torch.arange(1 / max_num_patches_per_side, 1.0, 1 / max_num_patches_per_side)
    fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / num_patches_h)
    fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / num_patches_w)
    bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
    bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)
    pos_ids = (bucket_coords_h[:, None] * max_num_patches_per_side + bucket_coords_w).flatten()
    return pos_ids



def get_rope_index_image_3D_dino(
    image_grid_thw: torch.LongTensor,  # Shape: (3,) for single image (t, h, w)
    curr_position_id: int,  # Base position ID for the image
    device=None, 
):
    """
    Calculate 3D RoPE indices for a single image.
    
    Args:
        image_grid_thw: Temporal, height, width dimensions of the image grid
        curr_position_id: Base position ID for the image's position sequence
        
    Returns:
        position_ids: 3D position indices (t, h, w) with shape (3, 1, num_img_tokens)
        mrope_position_deltas: Position delta tensor with shape (1, 1)
    """
    # Get spatial merge size from config
    # spatial_merge_size = 1 #self.config.vision_config.spatial_merge_size
    
    # Extract image dimensions (temporal, height, width)
    t, h, w = image_grid_thw  # Unpack from shape (3,)
    llm_grid_t = t.item()
    llm_grid_h = h.item() 
    llm_grid_w = w.item() 
    
    # Calculate total number of image tokens
    num_img_tokens = llm_grid_t * llm_grid_h * llm_grid_w # for img start end tokens
    
    # Generate 3D position indices for image tokens
    t_index = torch.arange(llm_grid_t, device=image_grid_thw.device)\
                .view(-1, 1)\
                .expand(-1, llm_grid_h * llm_grid_w)\
                .flatten()
    
    h_index = torch.arange(llm_grid_h, device=image_grid_thw.device)\
                .view(1, -1, 1)\
                .expand(llm_grid_t, -1, llm_grid_w)\
                .flatten()
    
    w_index = torch.arange(llm_grid_w, device=image_grid_thw.device)\
                .view(1, 1, -1)\
                .expand(llm_grid_t, llm_grid_h, -1)\
                .flatten()
    
    # Apply base position offset
    t_index = t_index + curr_position_id
    h_index = h_index + curr_position_id
    w_index = w_index + curr_position_id
    
    # Stack into (3, num_img_tokens) and add batch dimension
    position_ids = torch.stack([t_index, h_index, w_index], dim=0) #.unsqueeze(1)  # Final shape: (3, 1, num_img_tokens)
    
    # Calculate position delta
    max_position = position_ids.max()
    mrope_position_deltas = (max_position + 1 - num_img_tokens)\
                             .unsqueeze(0)\
                             .unsqueeze(1)  # Shape: (1, 1)
    my_delta = max_position - position_ids.min()
    # return position_ids.to(device), mrope_position_deltas.to(device)
    return position_ids, my_delta 




def get_rope_index_image_3D(
    image_grid_thw: torch.LongTensor,  # Shape: (3,) for single image (t, h, w)
    curr_position_id: int,  # Base position ID for the image
    device=None, 
):
    """
    Calculate 3D RoPE indices for a single image.
    
    Args:
        image_grid_thw: Temporal, height, width dimensions of the image grid
        curr_position_id: Base position ID for the image's position sequence
        
    Returns:
        position_ids: 3D position indices (t, h, w) with shape (3, 1, num_img_tokens)
        mrope_position_deltas: Position delta tensor with shape (1, 1)
    """
    # Get spatial merge size from config
    spatial_merge_size = 1 #self.config.vision_config.spatial_merge_size
    
    # Extract image dimensions (temporal, height, width)
    t, h, w = image_grid_thw  # Unpack from shape (3,)
    llm_grid_t = t.item()
    llm_grid_h = h.item() // spatial_merge_size
    llm_grid_w = w.item() // spatial_merge_size
    
    # Calculate total number of image tokens
    num_img_tokens = llm_grid_t * llm_grid_h * llm_grid_w # for img start end tokens
    
    # Generate 3D position indices for image tokens
    t_index = torch.arange(llm_grid_t, device=image_grid_thw.device)\
                .view(-1, 1)\
                .expand(-1, llm_grid_h * llm_grid_w)\
                .flatten()
    
    h_index = torch.arange(llm_grid_h, device=image_grid_thw.device)\
                .view(1, -1, 1)\
                .expand(llm_grid_t, -1, llm_grid_w)\
                .flatten()
    
    w_index = torch.arange(llm_grid_w, device=image_grid_thw.device)\
                .view(1, 1, -1)\
                .expand(llm_grid_t, llm_grid_h, -1)\
                .flatten()
    
    # Apply base position offset
    t_index = t_index + curr_position_id
    h_index = h_index + curr_position_id
    w_index = w_index + curr_position_id
    
    # Stack into (3, num_img_tokens) and add batch dimension
    position_ids = torch.stack([t_index, h_index, w_index], dim=0) #.unsqueeze(1)  # Final shape: (3, 1, num_img_tokens)
    
    # Calculate position delta
    max_position = position_ids.max()
    mrope_position_deltas = (max_position + 1 - num_img_tokens)\
                             .unsqueeze(0)\
                             .unsqueeze(1)  # Shape: (1, 1)
    my_delta = max_position - position_ids.min()
    # return position_ids.to(device), mrope_position_deltas.to(device)
    return position_ids, my_delta 



def prepare_attention_mask_per_sample(split_lens, attn_modes, device="cpu"):
    """
    nested_split_lens: A list of N lists of ints. Each int indicates the length of a split within 
        a sample, where each sample contains multiple splits with different attn modes.
    nested_attn_modes: whether to use full attn in each split.
    """
    sample_len = sum(split_lens)
    attention_mask = torch.zeros((sample_len, sample_len), dtype=torch.bool, device=device)

    csum = 0
    for s, attn_mode in zip(split_lens, attn_modes):
        assert attn_mode in ['causal', 'full', 'noise']
        if attn_mode == "causal":
            attention_mask[csum:csum + s, csum:csum + s] = torch.ones((s, s), device=device).tril()
            attention_mask[csum:csum + s, :csum] = 1
        else:
            attention_mask[csum:csum + s, csum:csum + s] = torch.ones((s, s))
            attention_mask[csum:csum + s, :csum] = 1
        csum += s

    csum = 0
    # noise mask: they are seperate
    # noise 1 only attend to noise 1, noise 2 to 2
    # full 1 to 1, 2 to 1 and 2. 
    for s, attn_mode in zip(split_lens, attn_modes):
        if attn_mode == "noise":
            attention_mask[:, csum : csum + s] = torch.zeros((sample_len, s))
            attention_mask[csum : csum + s, csum : csum + s] = torch.ones((s, s))
        csum += s

    attention_mask = torch.zeros_like(attention_mask, dtype=torch.float).masked_fill_(
        ~attention_mask, float("-inf")
    )

    return attention_mask


def split_integer_exp_decay(S, ng_sample_decay=1.0):
    if ng_sample_decay == 1.0:
        N = random.randint(1, S)
    else:
        base = (1 - ng_sample_decay) / (1 - math.pow(ng_sample_decay, S))
        p = [base * math.pow(ng_sample_decay, i) for i in range(S)]
        N = random.choices(list(range(1, S + 1)), p, k=1)[0]
    cumsum = [0] + sorted(random.sample(range(1, S), N - 1)) + [S]
    result = [cumsum[i+1] - cumsum[i] for i in range(len(cumsum) - 1)]
    return result, cumsum


def pil_img2rgb(image):
    if image.mode == "RGBA" or image.info.get("transparency", None) is not None:
        image = image.convert("RGBA")
        white = Image.new(mode="RGB", size=image.size, color=(255, 255, 255))
        white.paste(image, mask=image.split()[3])
        image = white
    else:
        image = image.convert("RGB")

    return image


def pil_img2rgb_no_alpha(image):
    if image.mode == "RGBA" or image.info.get("transparency", None) is not None:
        image = image.convert("RGBA")
        white = Image.new(mode="RGB", size=image.size, color=(255, 255, 255))
        white.paste(image, mask=image.split()[3])
        image = white
    else:
        image = image.convert("RGB")

    return image


def add_special_tokens(tokenizer):
    all_special_tokens = []
    for k, v in tokenizer.special_tokens_map.items():
        if isinstance(v, str):
            all_special_tokens.append(v)
        elif isinstance(v, list):
            all_special_tokens += v

    new_tokens = []

    if '<|im_start|>' not in all_special_tokens:
        new_tokens.append('<|im_start|>')

    if '<|im_end|>' not in all_special_tokens:
        new_tokens.append('<|im_end|>')

    if '<|vision_start|>' not in all_special_tokens:
        new_tokens.append('<|vision_start|>')

    if '<|vision_end|>' not in all_special_tokens:
        new_tokens.append('<|vision_end|>')

    num_new_tokens = tokenizer.add_tokens(new_tokens)
    bos_token_id = tokenizer.convert_tokens_to_ids('<|im_start|>')
    eos_token_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
    start_of_image = tokenizer.convert_tokens_to_ids('<|vision_start|>')
    end_of_image = tokenizer.convert_tokens_to_ids('<|vision_end|>')

    new_token_ids = dict(
        bos_token_id=bos_token_id, 
        eos_token_id=eos_token_id, 
        start_of_image=start_of_image, 
        end_of_image=end_of_image, 
    )

    return tokenizer, new_token_ids, num_new_tokens


def len2weight(x, loss_reduction='square'):
    if x == 0:
        return x
    if loss_reduction == 'token':
        return 1
    if loss_reduction == 'sample':
        return 1 / x
    if loss_reduction == 'square':
        return 1 / (x ** 0.5)
    raise NotImplementedError(loss_reduction)


def apply_template_qwenvl2_reconThenUnd(question_with_image_tokens,answer):
    chat_template1 = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'
    chat_template2=question_with_image_tokens
    chat_template3= '<|im_end|>\n<|im_start|>assistant'
    if len(answer)>0:
        chat_template4= '\n'+answer
    ret = []
    pattern = r'(<vit_image>|<dino_image>)'
    chat_template2_split = re.split(pattern, chat_template2)
    chat_template2_split = [p for p in chat_template2_split if len(p)>0]
    ret.append({
        'type':'text',
        'loss':False,
        'value':chat_template1,
    })
    ret.append({
        'type':'text',
        'loss':False,
        'value':'Reconstruct the 3D scene.',
    }) 
    
    for split_ in chat_template2_split:
        if split_ not in ['<vit_image>','<dino_image>']:
            ret.append({
                'type':'text',
                'loss':False,
                'value':split_,
            })
        elif split_=='<vit_image>':
            ret.append({
                'type':'vit',
                'loss':False,
                'value':split_,
            })
        elif split_=='<dino_image>':
            ret.append({
                'type':'dino',
                'loss':False,
                'value':split_,
            })
    ret.append(
        {
            'type':'text',
            'loss':False,
            'value':chat_template3,
        }
    )
    if len(answer)>0:
        ret.append(
            {
                'type':'text',
                'loss':True,
                'value':chat_template4,
            }
        )
    return ret
            

def apply_template_qwenvl2(question_with_image_tokens,answer):
    chat_template1 = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'
    chat_template2=question_with_image_tokens
    chat_template3= '<|im_end|>\n<|im_start|>assistant'
    if len(answer)>0:
        chat_template4= '\n'+answer
    ret = []
    pattern = r'(<vit_image>|<dino_image>)'
    chat_template2_split = re.split(pattern, chat_template2)
    chat_template2_split = [p for p in chat_template2_split if len(p)>0]
    ret.append({
        'type':'text',
        'loss':False,
        'value':chat_template1,
    })
    
    for split_ in chat_template2_split:
        if split_ not in ['<vit_image>','<dino_image>']:
            ret.append({
                'type':'text',
                'loss':False,
                'value':split_,
            })
        elif split_=='<vit_image>':
            ret.append({
                'type':'vit',
                'loss':False,
                'value':split_,
            })
        elif split_=='<dino_image>':
            ret.append({
                'type':'dino',
                'loss':False,
                'value':split_,
            })
    ret.append(
        {
            'type':'text',
            'loss':False,
            'value':chat_template3,
        }
    )
    if len(answer)>0:
        ret.append(
            {
                'type':'text',
                'loss':True,
                'value':chat_template4,
            }
        )
    return ret
            