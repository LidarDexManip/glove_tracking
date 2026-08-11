# Prompt for Codex on the second server

Copy the text below into Codex on the second server. Replace only paths that do
not exist there; keep the model, chunking, prompt selection, label mapping, and
SQLite format unchanged.

---

你正在一台有 NVIDIA GPU 的 Linux 服务器上工作。请直接完成下面的任务，不要只给我命令或计划。在执行较大的网络下载或磁盘写入前，先报告空间、GPU、CPU 和时间估算，然后继续执行。服务器是共享的，请使用单个 GPU worker、低 CPU/I/O 优先级，并保证任务中断后可以恢复。

目标：从 Hugging Face 下载已经用官方 Project Aria API 生成的 1408×1408 pinhole HOT3D Aria 数据集，然后使用仓库中最终版的 SAM2.1 Hiera Large 原生 SAM2VideoPredictor 流程，对全部 sequence 生成与原服务器相同结构的手部分割 mask 数据库。

## 1. 获取固定分支的代码

~~~bash
git clone --recurse-submodules https://github.com/LidarDexManip/glove_tracking.git
cd glove_tracking
git checkout agent/reproduce-hugg-aria-sam2
git submodule update --init --recursive
~~~

最终流程只使用这些入口：

- segment_hugg_aria_pinhole.py：单 sequence 推理。
- run_hugg_aria_sam2_batch.py：全部 sequence 顺序、可恢复推理。
- read_sam_mask_sqlite.py：导出单帧标签 mask。
- visualize_sam_mask_sqlite.py：生成 overlay PNG/MP4。

不要改回逐帧 image predictor、上一帧 logits prompt、YOLO detector，或旧的 simple/robust 脚本。

## 2. 检查资源并安装相同环境

先运行 nvidia-smi、df -h、free -h 和 lscpu。完整数据约 47 GB，最终 mask 约 8 GB，另预留至少 20 GB 临时与环境空间。参考环境：

- Python 3.10.20
- PyTorch 2.11.0+cu128 / torchvision 0.26.0
- NumPy 2.2.6
- OpenCV 4.13.0
- SAM-2 1.0
- projectaria-tools 1.5.1
- HOT3D submodule commit 240430b4ef43c87d19403ba0e9bfe94bcf41477f

~~~bash
conda env create -f environment.glove-hot3d.yml
conda run -n glove-hot3d python -m pip install -r requirements.glove-hot3d-gpu.txt
conda run -n glove-hot3d python -m pip install -r requirements.glove-hot3d.txt
conda run -n glove-hot3d python -m pip install "SAM-2==1.0" "projectaria-tools==1.5.1"
~~~

## 3. 下载处理好的 pinhole 数据集

先选择大容量磁盘，把下面的 /mnt/storage/USERNAME 替换为实际路径：

~~~bash
mkdir -p /mnt/storage/USERNAME/HUGG_ARIA_PINHOLE
ionice -c 2 -n 7 nice -n 10 hf download \
  LIDAR-GT/HUGG_ARIA_PINHOLE \
  --repo-type dataset \
  --local-dir /mnt/storage/USERNAME/HUGG_ARIA_PINHOLE \
  --max-workers 2
~~~

下载可以重复执行并恢复。完成后确认有且仅有 198 个 SEQUENCE/_SUCCESS.json，总视频帧数应为 701,519。不要下载原始 fisheye LIDAR-GT/HUGG_ARIA，也不要再次 undistort。

## 4. 准备 MANO 和 SAM2.1 Large 权重

MANO v1.2 受单独许可证约束，不能由脚本公开下载。请让我提供、或从我已获许可的 MANO 下载中放置：

~~~text
mano_v1_2/models/MANO_LEFT.pkl
mano_v1_2/models/MANO_RIGHT.pkl
~~~

如果文件缺失就在这里暂停并明确告诉我；不要用其他 MANO 版本代替。这两个文件总计约 7.4 MB，可以从旧服务器单独传输。

从 Meta 官方下载 SAM2.1 Hiera Large：

~~~bash
mkdir -p model/sam2
curl -fL --retry 5 \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt \
  -o model/sam2/sam2.1_hiera_large.pt
sha256sum model/sam2/sam2.1_hiera_large.pt
~~~

必须得到：

~~~text
2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318
~~~

## 5. 完全按照原服务器逻辑推理

逻辑不可擅自改变：每 300 帧作为一个 chunk；在靠近 chunk 中心的位置扫描 MANO mesh 投影，选择同时可见双手且可见顶点充足的帧；从 MANO 像素投影 extrema 产生左右手 bbox，并扩大 8%；以左右手作为两个 object，在一个 prompt frame 上调用 SAM2 原生 add_new_points_or_box；随后分别向前和向后调用 propagate_in_video。模型使用 bfloat16，视频帧 offload 到 CPU。每个 chunk 完成后事务提交 SQLite，因此中断后重跑同一命令即可恢复。

选择 6 个当前空闲 CPU 核。下面的 12-17 只是示例，应先根据 lscpu 和当前进程调整。只运行一个 batch worker，并把路径替换为实际路径：

~~~bash
mkdir -p /mnt/storage/USERNAME/sam2_hugg_aria_masks
tmux new-session -d -s hugg_sam2 \
  "cd /PATH/TO/glove_tracking && exec taskset -c 12-17 \
   ionice -c 2 -n 7 nice -n 10 env TQDM_DISABLE=1 \
   conda run --no-capture-output -n glove-hot3d \
   python run_hugg_aria_sam2_batch.py \
   --input-root /mnt/storage/USERNAME/HUGG_ARIA_PINHOLE \
   --output-root /mnt/storage/USERNAME/sam2_hugg_aria_masks"
~~~

处理过程中检查：

~~~bash
tmux has-session -t hugg_sam2
tail -n 30 /mnt/storage/USERNAME/sam2_hugg_aria_masks/batch.log
find /mnt/storage/USERNAME/sam2_hugg_aria_masks \
  -mindepth 2 -maxdepth 2 -name _SUCCESS.json | wc -l
find /mnt/storage/USERNAME/sam2_hugg_aria_masks \
  -mindepth 2 -maxdepth 2 -name masks.sqlite.partial | wc -l
nvidia-smi
~~~

不要并发跑多个 sequence。原服务器 RTX PRO 6000 Blackwell 上峰值显存约 1.8 GiB；701,519 帧整体用时约 9 小时 40 分，其他 GPU 会依性能变化。不同 GPU/PyTorch 架构可能造成极少量边界像素差异，但流程、标签和数据库结构必须相同。

## 6. 产物格式和最终验收

每个 sequence 输出：

~~~text
<MASK_ROOT>/<SEQUENCE>/
  masks.sqlite
  _SUCCESS.json
  prompt_frames/*.jpg
~~~

SQLite 每帧保存一张 zlib-compressed row-major uint8 label image：0=background、1=left_hand、2=right_hand。不要导出数十万张 PNG，需要时再按帧读取。

预期最终统计：

- 198/198 sequence 有 _SUCCESS.json
- 0 个 masks.sqlite.partial
- 701,519 个 frame row
- 2443 个 chunk，其中 1652 个 complete，791 个 no_prompt
- 约 8 GB 输出
- 62 个官方 test-set sequence 的 MANO 文件为空，因此其 mask 全零是预期行为；不要伪造 GT 或把它报告为推理失败

抽查训练 sequence：

~~~bash
conda run -n glove-hot3d python read_sam_mask_sqlite.py \
  /mnt/storage/USERNAME/sam2_hugg_aria_masks/P0002_59a84a3a/masks.sqlite \
  150 --output /tmp/P0002_frame_150_mask.png

conda run -n glove-hot3d python visualize_sam_mask_sqlite.py \
  P0002_59a84a3a \
  --dataset-root /mnt/storage/USERNAME/HUGG_ARIA_PINHOLE \
  --mask-root /mnt/storage/USERNAME/sam2_hugg_aria_masks \
  --start 0 --max-frames 300 --scale 0.5 \
  --output /tmp/P0002_overlay_10s.mp4
~~~

最后向我报告：数据下载路径和大小、环境与 checkpoint 校验、有效 sequence/帧数、SAM/no-prompt chunk 数、失败列表、总耗时、峰值 GPU、输出路径与大小。不要上传 mask，除非我之后明确要求。

---
