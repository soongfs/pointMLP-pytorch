# BUPT 机器学习课程作业 · 赛道一：ModelNet40 三维点云分类

组员：宋旗舰 2023210797 · 付玉莹 2023212832 · 付思语 2023210909

本项目基于 [PointMLP](https://github.com/ma-xu/pointMLP-pytorch)（ICLR 2022），
保留官方网络架构，新增对课程 normal-resampled txt 数据格式的训练、评估和现场预测能力。

## 1. 结果

官方 ModelNet40 测试集（2468 个样本，xyz-only，1024 点）：

| 推理方式 | Instance Accuracy | Class Accuracy |
|---|---|---|
| 单次确定性采样 | 93.03% | 90.16% |
| 10 次随机采样 voting | **93.40%** | **90.73%** |
| 20 次随机采样 voting | 93.44% | 90.56% |

两条加分线（Instance ≥ 92%、Class ≥ 90%）均已达到。现场预测推荐
`--num_votes 10 --vote_sampling random`。

## 2. 环境准备（uv）

依赖已用 [uv](https://docs.astral.sh/uv/) 管理。在项目根目录：

```bash
uv sync
```

### pointnet2_ops（可选，加速 FPS）

官方 PointMLP 用 `pointnet2_ops` 的 CUDA 算子做最远点采样（FPS）。当机器
`nvcc` 的 CUDA 版本与已安装的 PyTorch 编译版本匹配时，可编译安装：

```bash
export CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
bash scripts/install_pointnet2_ops.sh
```

若 CUDA 版本不匹配（例如 nvcc 为 CUDA 13.x、PyTorch 为 cu124）导致编译失败，
**无需处理**：模型内置纯 PyTorch FPS 回退路径，会自动启用，只是速度略慢，不影响精度。

## 3. 数据准备

课程数据为 normal-resampled txt 格式：`<class>/<sample_id>.txt`，每个样本
10000 行、每行 6 列 `x,y,z,nx,ny,nz`。本项目从 Hugging Face 的
`Pointcept/modelnet40_normal_resampled-compressed` 下载同结构数据：

```bash
cd classification_ModelNet40
HF_ENDPOINT=https://hf-mirror.com uv run python download_pointcept_modelnet40.py \
  --output data/modelnet40_normal_resampled
# 下载得到 .tar.gz 后解压到同目录
cd data/modelnet40_normal_resampled && tar -xzf *.tar.gz && cd -
```

数据目录若含官方 `modelnet40_train.txt` / `modelnet40_test.txt`，会自动按官方
split 划分（train 9843 / test 2468）；否则回退到按类别 80/20 划分。

## 4. 训练

```bash
cd classification_ModelNet40
CUDA_VISIBLE_DEVICES=0 uv run python main.py \
  --model pointMLP \
  --data_format txt \
  --data_root data/modelnet40_normal_resampled \
  --batch_size 32 --epoch 300 --num_points 1024 \
  --learning_rate 0.1 --min_lr 0.005 --weight_decay 0.0002 \
  --workers 8 --preload \
  --msg pointcept-xyz-s2026 --seed 2026
```

权重输出到 `checkpoints/pointMLP-pointcept-xyz-s2026-2026/best_checkpoint.pth`。
`--preload` 在启动时一次性把点云读入内存，加速多 epoch 训练（约需数 GB 内存）。

## 5. 评估（验证 voting 策略）

```bash
CUDA_VISIBLE_DEVICES=0 uv run python evaluate_course.py \
  --model pointMLP \
  --checkpoint checkpoints/pointMLP-pointcept-xyz-s2026-2026/best_checkpoint.pth \
  --data_root data/modelnet40_normal_resampled \
  --split test --num_points 1024 --batch_size 32 \
  --num_votes 10 --vote_sampling random
```

输出 Instance / Class Accuracy。支持重复 `--checkpoint` 做多模型 ensemble。

## 6. 现场预测（生成提交 CSV）

对下发的无标签测试集（目录或 zip 均可）生成提交文件：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python predict_course.py \
  --model pointMLP \
  --checkpoint checkpoints/pointMLP-pointcept-xyz-s2026-2026/best_checkpoint.pth \
  --test_dir /path/to/onsite_test \
  --output 赛道1-宋旗舰2023210797-付玉莹2023212832-付思语2023210909.csv \
  --num_points 1024 --batch_size 32 \
  --num_votes 10 --vote_sampling random
```

CSV 默认无表头，每行 `id,预测类别`，例如 `airplane_0627,airplane`。
加 `--header` 可写表头。重复 `--checkpoint` 可做多模型 ensemble。

## 7. 新增文件清单

| 文件 | 作用 |
|---|---|
| `course_train_data.py` | 有标签训练 Dataset，支持官方 split / 目录 / zip / preload |
| `course_data.py` | 无标签现场预测 Dataset，自动跳过元数据文件 |
| `predict_course.py` | 生成提交 CSV，支持 voting 与多模型 ensemble |
| `evaluate_course.py` | 在有标签 split 上评估，验证 voting 效果 |
| `modelnet40_classes.py` | 40 类标签顺序与名称映射 |
| `download_pointcept_modelnet40.py` | 从 Hugging Face 下载数据 |
| `models/pointmlp.py` | 新增 pointnet2_ops 缺失时的纯 PyTorch FPS 回退 |
| `main.py` | 新增 `--data_format txt` 等参数，接入课程数据 |

## 8. 设计思路

一页纸设计思路（中文 LaTeX，源码 `design.tex`，编译产物 `design.pdf`）
随最终交付材料单独提交，不纳入本代码仓库。
