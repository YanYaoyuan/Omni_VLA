import os
import json
import argparse
from tqdm.auto import tqdm
import numpy as np
import os.path as osp
from scipy import interpolate
from data.dataset_utils_vggt import *
import cv2
import decimal 

def main(args):
    data_root = "path to :s3://TartanAir/"
    conf_path = '~/petreloss.conf'
    client = Client(conf_path)
    
    sequences = []
    for seq in list(client.list(data_root)):
        names = client.list(os.path.join(data_root, seq, seq, 'Easy'))
        seq_ = [(seq, 'Easy', name) for name in names]
        sequences.extend(seq_)
        names = client.list(os.path.join(data_root, seq, seq, 'Hard'))
        seq_ = [(seq, 'Hard', name) for name in names]
        sequences.extend(seq_)
  
    out_data = []
    for seq in tqdm(list(sequences)):
        rgb_dir = os.path.join(data_root, seq[0], seq[0], seq[1], seq[2], 'image_left')
        num_imgs = len(list(client.list(rgb_dir)))
        
        cur_dict = {
            'seq_name': seq,
            'scene_name': 'tartanair',
            'num_images': num_imgs }
        out_data.append(cur_dict)
        
    print(f"{len(out_data)} valid sequences processed...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(os.path.join(args.output_dir, f"tartanair_recon_ann.jsonl"), 'w') as f:
        for item in out_data:
            f.write(json.dumps(item) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess HyperSim dataset')
    parser.add_argument('--output_dir', type=str, default="todo", help='Output directory')
    parser.add_argument('--min_num_images', type=int, default=24, help='Minimum number of images per sequence')
    parser.add_argument('--top_n', type=int, default=256, help='Number of similar frames to keep for each image')
    
    args = parser.parse_args()
    main(args)    