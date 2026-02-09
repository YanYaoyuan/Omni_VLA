# LIBERO Benchmark

This example runs the LIBERO benchmark: https://github.com/Lifelong-Robot-Learning/LIBERO

Note: When updating requirements.txt in this directory, there is an additional flag `--extra-index-url https://download.pytorch.org/whl/cu113` that must be added to the `uv pip compile` command.

This example requires git submodules to be initialized. Don't forget to run:

```bash
git submodule update --init --recursive
```

## With Docker (recommended)

```bash
# Grant access to the X11 server:
sudo xhost +local:docker

# To run with the default checkpoint and task suite:
SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build
```

You can customize the loaded checkpoint by providing additional `SERVER_ARGS` (see `scripts/serve_policy.py`), and the LIBERO task suite by providing additional `CLIENT_ARGS` (see `examples/libero/main.py`).
For example:

```bash
# To load a custom checkpoint (located in the top-level openpi/ directory):
export SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero --policy.dir ./my_custom_checkpoint"

# To run the libero_10 task suite:
export CLIENT_ARGS="--args.task-suite-name libero_10"
```

## Without Docker (not recommended)

Terminal window 1:

```bash
# Create virtual environment
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

# Run the simulation
python examples/libero/main.py

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx python examples/libero/main.py
```

Terminal window 2:

```bash
# Run the server
uv run --no-sync scripts/serve_policy.py --env LIBERO
```

## 使用本地 LIBERO 数据 + PyTorch 训练

若希望用**本地的 LIBERO 数据集**在 openpi 里做 **PyTorch 微调**，按下面步骤操作。

### 1. 准备原始 LIBERO 数据（RLDS 格式）

原始数据来源二选一：

- **从 HuggingFace 下载**： [openvla/modified_libero_rlds](https://huggingface.co/datasets/openvla/modified_libero_rlds)  
  - 用 `tensorflow_datasets` 加载时会自动下载到 TFDS 缓存目录（如 `~/tensorflow_datasets`）。
- **已有本地 RLDS 目录**：若已下载到某目录，记下该路径作为 `--data_dir`。

### 2. 转换为 LeRobot 格式

安装转换依赖：

```bash
uv pip install tensorflow tensorflow_datasets
```

（可选）若希望训练时用「本地 LeRobot 目录」而不是 HuggingFace Hub，在 `examples/libero/convert_libero_data_to_lerobot.py` 里把 `REPO_NAME` 改为本地用的名字，例如：

```python
REPO_NAME = "libero"  # 转换后的数据会放在 $HF_LEROBOT_HOME/libero
```

指定 **LeRobot 数据根目录** 并执行转换（`--data_dir` 为 TFDS 数据目录，不填则用默认缓存）：

```bash
export HF_LEROBOT_HOME=/path/to/your/lerobot/data   # 转换后的输出目录
uv run --no-sync examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/tfds/data
```

转换完成后，LeRobot 数据集位于：`$HF_LEROBOT_HOME/<REPO_NAME>`（例如 `$HF_LEROBOT_HOME/libero`）。

### 3. 计算归一化统计量

使用「本地 LIBERO」对应的训练配置（若你上面用了 `REPO_NAME="libero"`，则用 `pi05_libero_local`）：

```bash
export HF_LEROBOT_HOME=/path/to/your/lerobot/data
uv run --no-sync scripts/compute_norm_stats.py --config-name pi05_libero_local
```

若你自定义了 `REPO_NAME`（例如 `my_name/libero`），需在 `src/openpi/training/config.py` 里新增或修改一个 `TrainConfig`，使其 `data.repo_id` 与 `REPO_NAME` 一致，再用该 config 名运行上述命令。

### 4. PyTorch 训练

保持同一 `HF_LEROBOT_HOME`，用对应 config 启动训练（单卡示例）：

```bash
export HF_LEROBOT_HOME=/path/to/your/lerobot/data
uv run --no-sync scripts/train_pytorch.py pi05_libero_local --exp_name my_libero_run --save_interval 5000
```

多卡（例如 2 卡）：

```bash
export HF_LEROBOT_HOME=/path/to/your/lerobot/data
torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi05_libero_local --exp_name my_libero_run --save_interval 5000
```

 checkpoint 会写在项目下的 `checkpoints/pi05_libero_local/my_libero_run/` 中（按 step 子目录保存）。

### 5. 用命令行覆盖 `repo_id`（可选）

若不想改 config 文件，也可以直接通过命令行覆盖数据源，例如使用已转换好的 `my_name/libero`：

```bash
export HF_LEROBOT_HOME=/path/to/your/lerobot/data
uv run --no-sync scripts/train_pytorch.py pi05_libero --exp_name my_run --data.repo_id my_name/libero
```

此时仍需先用同一 `repo_id` 跑一遍 `compute_norm_stats.py`（例如 `--config-name pi05_libero` 并加上 `--data.repo_id my_name/libero`，若 tyro 支持该覆盖）。

---

## Results

If you want to reproduce the following numbers, you can evaluate the checkpoint at `gs://openpi-assets/checkpoints/pi05_libero/`. This
checkpoint was trained in openpi with the `pi05_libero` config.

| Model | Libero Spatial | Libero Object | Libero Goal | Libero 10 | Average |
|-------|---------------|---------------|-------------|-----------|---------|
| π0.5 @ 30k (finetuned) | 98.8 | 98.2 | 98.0 | 92.4 | 96.85
