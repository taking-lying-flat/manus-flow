# Manus Flow

基于 PyTorch 的 CIFAR-10 生成模型实现集合。所有图像训练入口共用根目录下的
`dataloader.py`，数据集固定为 CIFAR-10，不需要传入数据集名称或数据路径。

## 环境

推荐使用 Python 3.12 和 UV：

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install torch torchvision numpy scipy tqdm einops einx hyper-connections torchdiffeq
```

检查 PyTorch 和 CUDA：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

训练数据会自动下载到仓库根目录的 `data/`。首次运行需要网络连接。

## 项目结构

| 目录 | 实现 |
| --- | --- |
| `2013_kingma_welling_vae` | Bernoulli VAE |
| `2014_dinh_krueger_nice` | NICE |
| `2015_rezende_normalizing_flow` | 带条件平面流的 DLGM |
| `2015_sohl_dickstein_diffusion` | 多步扩散模型 |
| `2017_dinh_sohl_realnvp` | RealNVP |
| `2018_chen_torchdiffeq` | 本地 ODE 求解器 |
| `2018_kingma_dhariwal_glow` | Glow |
| `2019_song_ermon_ncsn` | NCSN |
| `2020_song_ermon_ncsnv2` | NCSNv2 |
| `2022_liu_rectified_flow` | Rectified Flow |
| `2023_lipman_flow_matching` | Flow Matching |

根目录公共模块：

- `dataloader.py`：CIFAR-10 下载、增强、反量化和 DataLoader。
- `training_utils.py`：日志、噪声尺度和图像网格工具。

## 训练

在仓库根目录运行：

```bash
python 2013_kingma_welling_vae/main.py
python 2014_dinh_krueger_nice/train.py
python 2015_rezende_normalizing_flow/train.py
python 2015_sohl_dickstein_diffusion/main.py
python 2017_dinh_sohl_realnvp/train.py
python 2018_kingma_dhariwal_glow/train.py
python 2019_song_ermon_ncsn/train.py
python 2020_song_ermon_ncsnv2/train.py
python 2022_liu_rectified_flow/train.py
python 2023_lipman_flow_matching/image_gen.py
```

除 VAE 外，各训练入口都可以查看命令行参数：

```bash
python 2014_dinh_krueger_nice/train.py --help
python 2020_song_ermon_ncsnv2/train.py --help
python 2022_liu_rectified_flow/train.py --help
```

VAE 的训练参数位于 `2013_kingma_welling_vae/main.py` 顶部。

## 输出

训练输出保存在对应模型目录的 `runs/cifar10/` 或 `output/cifar10/` 下。
训练入口默认保存生成样本；Rectified Flow 启用 `--use-consistency` 时会额外保存
包含 EMA 状态的 checkpoint。

## 本地 ODE 求解器

目录名以数字开头，建议通过 `importlib` 导入：

```python
import importlib

torchdiffeq_local = importlib.import_module("2018_chen_torchdiffeq")
odeint = torchdiffeq_local.odeint
```

## 代码检查

```bash
python -m compileall -q .
ruff check .
```

## License

见 `LICENSE`。
