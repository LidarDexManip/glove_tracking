# Baseline / Gaussian surface 视频对比网页

入口：[compare_hugg_restorers_web.py](../compare_hugg_restorers_web.py)。这是独立工具，不修改训练代码、训练数据或 checkpoint。

## 启动与访问

在本机现有环境中：

```bash
cd /home/szeng87/glove_tracking
source /data/szeng87/venvs/gaussian-hand/bin/activate
python compare_hugg_restorers_web.py --gpu 0 --port 7862
```

依赖沿用 diffusion 环境及 `requirements.web-ui.txt`（Gradio 6.20.0），同时需要 ffmpeg、ffprobe、nvidia-smi 和已缓存的 SD1.5 权重。GPU worker 强制离线，不自动下载模型。

默认仅监听服务器本机 `http://127.0.0.1:7862`。从自己的电脑访问时，使用 SSH 端口转发：

```bash
ssh -N -L 7862:127.0.0.1:7862 USER@SERVER
```

然后在电脑浏览器打开同一地址。没有公开分享链接。若必须使用 `--host 0.0.0.0`，程序要求设置 `HAND_COMPARE_USERNAME`、`HAND_COMPARE_PASSWORD`；仍建议 SSH 转发。

## 操作

1. 选择 sequence，以及自动 / 左手 / 右手 / 联合双手 crop。
2. 设置起始源帧、1–10 秒时长、采样 FPS；不确定起点时点击“寻找下一可用片段”。
3. 分别选择 baseline 和 Gaussian checkpoint，默认都是各自的 `controlnet_final.pt`（step 15380）。
4. 点击“生成同步对比视频”，在三个标签页观看或下载。

三个视频都是左右同步的三列，不依赖多个播放器同步：

| 标签页 | 左 | 中 | 右 |
|---|---|---|---|
| 原始模型输出 | 原图手部 crop | baseline 原始输出 | Gaussian 原始输出 |
| 各自输入 | 原图手部 crop | naive MANO 条件图 | Gaussian 条件图 |
| 贴回原视频 | 原始第一视角视频 | baseline 区域贴回 | Gaussian 区域贴回 |

“原始模型输出”不与目标图混合，是观察模型差异的主要视图。贴回视频只编辑所选 crop 的原有 edit mask；选单手时，另一只手仍是原视频，不代表另一只手也经过推理。全帧视图缩小到 512×512 展示，crop 视图保持模型的 512×512 分辨率。

默认每个内部画面顺时针旋转 90°；模型收到的输入和标题不会旋转。可以关闭该选项。

## 可比性边界

- Baseline 配置：`hugg_aria_glove2hand_faithful_mano_adaptive_splitcrop_v1.json`，输入来自 `data/HUGG_ARIA_MANO_ALIGNED`。
- Gaussian 配置：`hugg_aria_gaussian_surface_v2_4gpu.json`，输入来自 `outputs/gaussian_hand/diffusion_strict_v2/aligned`，保留 premultiplied RGB 处理。
- 两边均调用训练使用的 `make_hugg_aria_dataset`，不另写条件图逻辑。使用同一训练 manifest：136 sequences、524917 crop 记录。
- 每帧实际校验目标图、crop、MANO/condition/edit/loss/wrist/visible-hand/SAM-exclusion masks 一致，任一不一致立即失败。
- 相同 denoising steps、guidance、prompt、seed。每帧都重置相同 seed；没有时序模型、插帧或平滑。
- **这是训练集内、过拟合模型的视觉比较，不是未见数据泛化评估，也不是原训练随机过程的逐位重放。**
- 自动 crop 优先联合双手，否则选择 manifest 中面积较大的手；选择独立于模型输出，但可能随时间切换。若想固定手，请显式选左或右。
- 原视频 30 FPS；例如 5 FPS 输出每隔 6 个源帧推理一次，播放时长不变。5 秒 × 5 FPS 是每模型 25 帧；5 秒 × 30 FPS 是每模型 150 帧。
- 只接受所有采样帧均存在所选训练 crop 的窗口。不跳过缺帧、不使用空白 Gaussian 缓存、不按 Gaussian 拟合时的遮挡阈值再次筛选 diffusion 数据。窗口缺失时可换起点、缩短时长或换 crop 类型。

## 资源、缓存与复现

网页加载和选参数不占 GPU。生成时独立子进程在指定的一张 GPU 上顺序运行两个 checkpoint，batch=1，共用冻结骨干；结束后退出释放显存。默认 PyTorch 显存上限为该卡容量的 12%，可用 `--memory-fraction` 调整。

默认检测显存占用超过 1 GiB 或 GPU 利用率超过 10% 时拒绝加载模型，避免干扰现有任务；明确需要共享时可在启动命令加 `--allow-shared-gpu`，仍要求至少 16 GiB 可用显存。这是启动时的检查，不能预留 GPU 或阻止其他用户后来启动任务。

一次只运行一个对比任务，同用户、同 GPU 的多个本工具实例也使用文件锁互斥。取消按钮只终止本会话的推理进程组；不影响训练或其他用户进程。一个任务最多运行一小时。

已实测 GPU 0、两个 final checkpoint、1 秒 × 5 FPS、20 steps：10 次推理及编码约 14.3 秒（不含启动前哈希校验），峰值 PyTorch 显存约 6.6 GiB。同请求缓存返回约 2.4 秒。更长片段开销近似随帧数和 denoising steps 增长；默认 3 秒 × 5 FPS 通常预留约 1 分钟。

默认 3 秒 × 5 FPS 的完整网页接口测试已完成：每模型 15 帧，推理及编码约 31.3 秒，包含校验、加载及接口返回的总耗时约 40.8 秒；结束后 GPU 0 恢复到空闲。测试结果目录为 `P0001_f6cc0cc8_807d95bc06_rupwihaa`。全仓 CPU 回归测试为 76 passed、1 skipped（CUDA 测试在 CPU-only 验证中跳过）；另有上述真实 GPU 推理测试。

结果保存在 `outputs/hand_restoration/paired_video_web/<sequence>_<hash>_<unique>/`：

- 三个 MP4、`contact_sheet.jpg`、逐模型的条件图 / 原始输出 / 贴回帧。
- `manifest.json`：源帧及全局 manifest 索引、实际 checkpoint step/hash、配置 / 输入素材 / 代码哈希、参数、检查结果、计时和逐帧 PSNR。PSNR 仅作辅助，不替代视觉判断。
- `request.json`、`selected_frames.jsonl`、`worker.log`、`status.json`。
- `_SUCCESS.json`：完成标记和结果文件哈希；没有它的失败/取消任务不会用作缓存。

只有请求参数、checkpoint、配置、输入素材、相关代码都相同，且结果哈希检查通过，才复用结果。失败任务目录保留用于排错，不会自动删除用户文件。服务日志可自行重定向保存。

CPU 回归测试：

```bash
GRADIO_ANALYTICS_ENABLED=False CUDA_VISIBLE_DEVICES='' \
  python -m unittest discover -s tests -p test_paired_video.py -v
```
