"""Render quantitative evaluation artifacts; never loads models or uses a GPU."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def improvement(baseline, gaussian):
    return (baseline-gaussian)/baseline*100


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    report = json.loads((directory/'summary.json').read_text())
    provenance = json.loads((directory/'provenance.json').read_text())
    b, g = (report['overall'][arm] for arm in ('baseline','gaussian'))
    lines = ['# MANO baseline 与 Gaussian surface：手部定量比较', '',
        '本次结果：可见手部整体基本持平，没有明显、统一的量化优势。Gaussian 的平均 PSNR 与 MAE 略好，'
        'SSIM 几乎不变，但平均 MSE 变差；改善更明显的区域是手腕及被遮挡的 MANO 投影区域。', '',
        '## 评估口径', '',
        f"双方 final checkpoint，均为 step 15380；固定 seen-eval manifest 的全部 {report['n']} 个 crop，"
        f"覆盖 {len(report['subjects'])} 人、{report['sequences']} 个 sequence。全部样本属于训练集，"
        '衡量过拟合还原效果，不代表泛化。没有按本次输出质量挑选样本。', '',
        f"逐帧确认目标图、crop 和所有相关 masks 一致；双方使用各自的渲染输入。"
        f"FP32、batch=1、{provenance['inference_steps']} denoising steps、相同 seed={provenance['seed']}。"
        '指标均基于未贴回原图的 raw 模型输出，避免通过复制背景提高分数。', '',
        '主区域为 `visible_hand_mask = MANO ∩ SAM`。PSNR、MAE 按 RGB 手部像素计算，'
        '下表为逐 crop 等权平均；MAE 使用 0–255 色阶。SSIM 采用 RGB、11×11 Gaussian window、'
        'sigma=1.5，只平均窗口完全位于手部内部的位置，排除背景窗口。', '',
        '## 主结果：可见手部', '',
        '| 指标 | Baseline | Gaussian | 变化 |',
        '|---|---:|---:|---:|']
    br, gr = b['regions']['visible_hand'],g['regions']['visible_hand']
    for title,key in [('PSNR ↑（dB）','mean_psnr_db'),('MAE ↓（0–255）','mean_mae_255'),
                      ('MSE ↓（RGB 0–1）','mean_mse'),
                      ('手部内部 SSIM ↑','mean_ssim_interior')]:
        change = f"{gr[key]-br[key]:+.4f}"
        if key=='mean_mae_255':
            change=f"下降 {improvement(br[key],gr[key]):.2f}%"
        if key=='mean_mse':
            change=f"变化 {(gr[key]/br[key]-1)*100:+.2f}%（增加表示更差）"
        lines.append(f'| {title} | {br[key]:.6f} | {gr[key]:.6f} | {change} |')
    frames=[json.loads(line) for line in (directory/'baseline/frames.jsonl').open()]
    excluded=[r for r in frames if r['regions']['visible_hand'] is None]
    lines += ['',f"有效 crop：PSNR/MAE {br['n']}，SSIM {br['ssim_n']}。"
        f"像素合并 PSNR（大手占更大权重）为 {br['pooled_psnr_db']:.4f} → {gr['pooled_psnr_db']:.4f} dB。",'',
        f"可见手部 mask 为空的 {len(excluded)} 个 crop 在双方同时排除，不当作零误差： "
        + '; '.join(f"{r['sequence_id']}/{r['frame']}/{r['hand']}" for r in excluded) + '。', '',
        '## 按 subject 分组', '',
        '| Subject | crop 数 | Baseline PSNR | Gaussian PSNR | Δ PSNR | MAE 下降 |',
        '|---|---:|---:|---:|---:|---:|']
    for subject, arms in report['by_subject'].items():
        x,y=(arms[a]['regions']['visible_hand'] for a in ('baseline','gaussian'))
        lines.append(f"| {subject} | {x['n']} | {x['mean_psnr_db']:.3f} | {y['mean_psnr_db']:.3f} | "
                     f"{y['mean_psnr_db']-x['mean_psnr_db']:+.3f} | {improvement(x['mean_mae_255'],y['mean_mae_255']):.2f}% |")
    paired = report['visible_hand_paired_psnr']
    lines += ['',f"Gaussian 手部 PSNR 较高的 crop：{paired['positive_crops']}/{paired['total_crops']}；"
        f"sequence：{paired['positive_sequences']}/{paired['total_sequences']}。", '',
        'PSNR 是对 MSE 取对数后再平均，因此平均 PSNR 略升与平均 MSE 增大可以同时发生；'
        '不能只挑其中更有利的一项。按 sequence 重采样的探索性 PSNR 差值 95% 区间为 '
        f"[{paired['sequence_bootstrap_95ci'][0]:+.3f}, {paired['sequence_bootstrap_95ci'][1]:+.3f}] dB，"
        '包含零；该区间未处理同 subject 内 sequence 的相关性，不作为严格方法显著性检验。', '',
        '## 严格配对重算的 diffusion loss', '',
        f"每个 crop 使用 {provenance['noise_repeats']} 组确定的随机种子，双方的 VAE target sampling、"
        'noise、timestep 一致；相同 FP32、batch=1。手部 loss 使用训练的 MANO mask 下采样到 latent，'
        '不等于上述可见手部像素误差。Weighted loss 使用手部权重 10。', '',
        '| 指标 ↓ | Baseline | Gaussian | 下降 |', '|---|---:|---:|---:|']
    for key in ('hand_region_loss','weighted_loss','background_region_loss'):
        x,y=b['paired_noise_loss'][key],g['paired_noise_loss'][key]
        lines.append(f'| {key} | {x:.6f} | {y:.6f} | {improvement(x,y):.2f}% |')
    lines += ['', '## 其他区域（raw 输出）', '',
        '| 区域 | Baseline PSNR | Gaussian PSNR | MAE 下降 |', '|---|---:|---:|---:|']
    for name in ('mano','edit','wrist','occluded_mano'):
        x,y=b['regions'][name],g['regions'][name]
        lines.append(f"| {name} | {x['mean_psnr_db']:.3f} | {y['mean_psnr_db']:.3f} | "
                     f"{improvement(x['mean_mae_255'],y['mean_mae_255']):.2f}% |")
    lines += ['', '`mano` 包含被物体遮挡的投影位置；`edit` 包含扩张边界及手腕；'
        '`occluded_mano` 主要用于检查物体/遮挡区域是否被破坏，不是可见皮肤质量。', '',
        '## 原训练日志', '', '| 指标 ↓ | Baseline | Gaussian | 下降 |','|---|---:|---:|---:|']
    lb,lg=(report['training_logs'][a] for a in ('baseline','gaussian'))
    for group,key,title in [('last_1000','hand_region_loss','最后 1000 step 手部训练 loss 均值'),
                            ('last_1000','loss','最后 1000 step weighted training loss 均值'),
                            ('final_validation','validation_hand_region_loss','日志最终 validation 手部 loss')]:
        x,y=lb[group][key],lg[group][key]
        lines.append(f'| {title} | {x:.6f} | {y:.6f} | {improvement(x,y):.2f}% |')
    lines += ['', '原训练为 8 卡与 4 卡，每卡 batch 不同，尽管全局训练 batch 一致，'
        '随机噪声与时间步的逐样本对应不同。原 validation 每卡 batch 为 8 与 16，'
        '日志 loss 只作训练趋势参考，不作严格配对结论。', '',
        '![Training loss curves](loss_curves.png)', '', '## 限制与复现', '',
        '- 183 个固定 crop 覆盖所有 sequence，但仍不是全训练帧评估；同一人多个 sequence 有相关性。',
        '- 只比较一对训练 checkpoint 和一个生成 seed；不是多次独立训练实验，小幅差异不等于稳定的方法优势。',
        '- PSNR/MAE/SSIM 衡量还原误差和结构相似度，不能单独证明真实感、身份细节或时序稳定性。',
        '- SSIM 为排除背景而收缩窗口中心区域，因此不能完整覆盖比窗口更细的手指；PSNR/MAE 仍覆盖全部可见手部像素。',
        '- SAM/几何 mask 不是人工逐像素真值，沿用两边相同的近似以保证公平。',
        '- 原始帧级指标及 raw PNG 位于 baseline/、gaussian/；baseline/reference NPZ 保存目标及区域 masks。',
        '- provenance.json 保存 checkpoint / manifest / 配置 / 脚本哈希；summary.json 保存聚合数值。', '',
        f"本次计算耗时 {report['elapsed_seconds']/60:.2f} 分钟，峰值 PyTorch 显存 "
        f"{report['peak_gpu_bytes']/1024**3:.2f} GiB。", '',
        '复现（先选空闲卡；输出目录必须不存在，避免覆盖既有结果）：', '',
        '```bash',
        'CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \\',
        '  python -m scripts.evaluation.quantify_paired_restorers --output NEW_OUTPUT_DIR',
        'python -m scripts.evaluation.report_paired_restorers NEW_OUTPUT_DIR',
        '```', '']
    (directory/'REPORT.md').write_text('\n'.join(lines))
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for arm,title in [('baseline','MANO baseline'),('gaussian','Gaussian surface')]:
        rows=list(csv.DictReader(Path(report['training_logs'][arm]['path']).open()))
        steps=np.array([int(r['total_step']) for r in rows])
        values=np.array([float(r['hand_region_loss']) for r in rows])
        axes[0].plot(steps[999:],np.convolve(values,np.ones(1000)/1000,mode='valid'),label=title)
        validation=[r for r in rows if r['validation_hand_region_loss']]
        axes[1].plot([int(r['total_step']) for r in validation],
                     [float(r['validation_hand_region_loss']) for r in validation],'o-',label=title)
    axes[0].set_title('Training hand loss (1000-step rolling mean)')
    axes[1].set_title('Logged validation hand loss (unpaired noise)')
    for ax in axes:
        ax.set_xlabel('Optimizer step');ax.set_ylabel('Latent noise-prediction MSE')
        ax.legend();ax.grid(alpha=.25)
    fig.savefig(directory/'loss_curves.png',dpi=160)
    plt.close(fig)
    print(directory/'REPORT.md')


if __name__ == '__main__':
    main()
