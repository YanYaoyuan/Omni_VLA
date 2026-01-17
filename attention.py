import torch
import matplotlib.pyplot as plt
import numpy as np

def visualize_attention_mask():
    # 1. 模拟各个部分的长度
    L_prefix = 12  # 语言 + 初始图像 (Prefix)
    L_middle = 16  # 视频序列/Cosmos特征 (Middle)
    L_suffix = 8   # 机器人状态 + 噪声动作 (Suffix)
    
    N = L_prefix + L_middle + L_suffix
    
    # 2. 构造 1D att_masks 触发向量
    # 0 表示延续当前注意力块（双向），1 表示进入下一个因果步进
    att_masks_1d = torch.zeros(N, dtype=torch.long)
    
    # Prefix 部分：全为 0 (前缀全向关注)
    # Middle 部分：第一个 token 设为 1，开启因果流
    att_masks_1d[L_prefix] = 1 
    # Suffix 部分：第一个 token 设为 1，开启后续因果依赖
    att_masks_1d[L_prefix + L_middle] = 1
    
    # 3. 核心逻辑：计算 2D 掩码
    # cumsum 控制了“谁能看谁”：只有 cumsum 较小或相等的位置才能被看见
    cumsum = torch.cumsum(att_masks_1d, dim=0).unsqueeze(0)
    att_2d_mask = (cumsum.T >= cumsum).to(torch.float32)

    # 4. 绘图
    plt.figure(figsize=(10, 8), dpi=120)
    plt.imshow(att_2d_mask, cmap='Blues', interpolation='nearest')
    
    # 添加区域分割线
    plt.axvline(x=L_prefix - 0.5, color='red', linestyle='--', alpha=0.6)
    plt.axvline(x=L_prefix + L_middle - 0.5, color='red', linestyle='--', alpha=0.6)
    plt.axhline(y=L_prefix - 0.5, color='red', linestyle='--', alpha=0.6)
    plt.axhline(y=L_prefix + L_middle - 0.5, color='red', linestyle='--', alpha=0.6)

    # 标注文本
    plt.text(L_prefix/2, -1, 'Prefix\n(Lang+Img)', ha='center', va='bottom', fontsize=10)
    plt.text(L_prefix + L_middle/2, -1, 'Middle\n(Cosmos)', ha='center', va='bottom', fontsize=10)
    plt.text(L_prefix + L_middle + L_suffix/2, -1, 'Suffix\n(State+Act)', ha='center', va='bottom', fontsize=10)

    plt.title("Hybrid Prefix-Causal Attention Mask Matrix", fontsize=14, pad=25)
    plt.xlabel("Key Indices (Context)", fontsize=12)
    plt.ylabel("Query Indices (Target)", fontsize=12)
    
    # 颜色条说明：1.0 可视，0.0 掩码
    cbar = plt.colorbar(ticks=[0, 1])
    cbar.ax.set_yticklabels(['Masked (0)', 'Attend (1)'])
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_attention_mask()