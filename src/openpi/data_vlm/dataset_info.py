from .interleave_datasets import UnifiedEditIterableDataset
from .vlm_dataset import SftJSONLIterableDataset
from .recon_dataset import SftJSONLIterableReconDataset
from .interleave_datasets.recon_then_und_dataset import ReconthenUndIterableDataset

DATASET_REGISTRY = {
    'vlm_sft': SftJSONLIterableDataset,
    'recon_then_und': ReconthenUndIterableDataset,
    'recon': SftJSONLIterableReconDataset, 
}

DATASET_INFO = {
	'vlm_sft':{
        'llava_ov': {
			'data_dir': 'your_data_path/g2vlm_example/vlm/images',
			'jsonl_path': 'your_data_path/g2vlm_example/vlm/llava_ov_si.jsonl',
			'num_total_samples': 1000
		},
    },

	'recon_then_und':{
		'spatial_data': {
			'data_dir': 'your_data_path/g2vlm_example/joint_trainng/images',
			'num_files': 10,
			'num_total_samples': 1000,
			"parquet_info_path": 'your_data_path/g2vlm_example/joint_trainng/parquet_info', # information of the parquet files
		},
	},
    
    'recon': {
		'scannet': {
			'data_dir': 'your_data_path/g2vlm_example/recon/images',
			'jsonl_path': 'your_data_path/g2vlm_example/recon/scannet.jsonl',
			'num_total_samples': 2000
		},
    },
}