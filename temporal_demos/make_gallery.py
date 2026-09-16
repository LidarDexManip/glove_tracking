"""Build a static local review gallery with client-side shortlist export."""
import argparse
import html
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args();root=args.directory.resolve()
    summary=json.loads((root/'summary.json').read_text())
    provenance=json.loads((root/'provenance.json').read_text())
    cards=[]
    for clip in summary['clips']:
        name=html.escape(clip['id'],quote=True)
        old,new=clip['metrics']['gaussian_only'],clip['metrics']['animatediff']
        key='flow_compensated_temporal_residual_mae_255'
        delta=(1-new[key]/old[key])*100
        warning=('<p style="color:#ffd08a">注意：这一段时序误差代理指标更低，但手部还原误差变大；保留作取舍对照，不优先推荐成片。</p>'
                 if new['visible_hand_psnr']<old['visible_hand_psnr']-.3 else '')
        cards.append(f'''<article id="{name}">
<h2><label><input type="checkbox" value="{name}" onchange="remember()"> {name}</label></h2>
<p>{clip['duration']:g} 秒 · {clip['hand']} hand · 源帧 {clip['source_start']} 起 · 可见比例最低 {clip['dense_visible_min']:.1%}</p>
<video controls muted loop playsinline preload="metadata" poster="{name}/contact_sheet.jpg" src="{name}/comparison.mp4"></video>
<nav><button onclick="switchVideo(this,'comparison.mp4')">四列对比</button>
<button onclick="switchVideo(this,'animatediff_crop.mp4')">AnimateDiff 手部</button>
<button onclick="switchVideo(this,'gaussian_only_crop.mp4')">不加 AnimateDiff</button>
<button onclick="switchVideo(this,'animatediff_full.mp4')">贴回原视频</button>
<a href="{name}/animatediff_full.mp4" download>下载贴回视频</a></nav>
<p class="metric">手部 PSNR {old['visible_hand_psnr']:.2f} → {new['visible_hand_psnr']:.2f} dB；时序误差下降 {delta:+.1f}%（光流代理指标，不能代替观看）。</p>{warning}
</article>''')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gaussian + AnimateDiff · Demo shortlist</title>
<style>body{max-width:1400px;margin:32px auto;padding:0 22px;font:16px/1.6 system-ui;background:#11151d;color:#e7edf6}h1{font-size:30px}h2{font-size:18px}p{color:#b8c3d4}article{background:#1b2230;padding:20px;border-radius:14px;margin:24px 0}video{width:100%;max-height:70vh;background:#080a0f;border-radius:8px}button,a{display:inline-block;background:#34435b;color:#eef4ff;border:0;padding:8px 14px;border-radius:7px;cursor:pointer;margin:6px 6px 0 0;text-decoration:none}input{width:20px;height:20px;vertical-align:middle}.toolbar{position:sticky;top:0;background:#11151dee;padding:10px;z-index:2}.metric{font-size:14px}</style>
<h1>Gaussian diffusion + AnimateDiff：候选 demo</h1>
<p>每段 5 秒、10 FPS，内部画面顺时针 90°；使用原 Gaussian-overlay final checkpoint，无额外训练。四列：原图 / Gaussian 条件 / 不加时序模块 / 加 AnimateDiff。此批按输入质量筛选，不能作为整体模型评测。</p>
<p>Motion strength 0.25；固定片段 crop，16 帧时序上下文、8 帧重叠。贴回视频仅修改所选手部的 edit mask，其他区域保留原视频。勾选喜欢的片段，导出选择清单。</p>
<div class="toolbar"><button onclick="downloadPicks()">导出所选清单 JSON</button><span id="count"></span></div>
<h2>全批快速浏览</h2><video controls muted loop playsinline preload="metadata" src="review_all.mp4"></video>
'''+'\n'.join(cards)+'''
<script>
const key='gaussian-animatediff-demo-v1-picks';
function checked(){return Array.from(document.querySelectorAll('input:checked')).map(x=>x.value)}
function remember(){try{localStorage.setItem(key,JSON.stringify(checked()))}catch(e){}document.querySelector('#count').textContent='已选 '+checked().length+' 段'}
function switchVideo(button,file){const a=button.closest('article'),v=a.querySelector('video');v.pause();v.src=a.id+'/'+file;v.removeAttribute('poster');v.load()}
function downloadPicks(){const data={selected:checked(),note:'Manual demo shortlist; model quality is not inferred from selected clips.'};const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='animatediff_demo_picks.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
try{const saved=JSON.parse(localStorage.getItem(key)||'[]');document.querySelectorAll('input').forEach(x=>x.checked=saved.includes(x.value))}catch(e){}remember();
</script></html>'''
    (root/'index.html').write_text(page)
    lines=['# AnimateDiff demo batch','',f"Checkpoint step: {provenance['checkpoint_step']}. Extra training: none.",
        f"GPU wall time: {summary['elapsed_seconds']:.1f}s. Peak PyTorch memory: {summary['peak_gpu_bytes']/2**30:.2f} GiB.",'',
        '| Clip | PSNR off | PSNR on | Temporal error off | Temporal error on |',
        '|---|---:|---:|---:|---:|']
    for clip in summary['clips']:
        b,g=clip['metrics']['gaussian_only'],clip['metrics']['animatediff'];key='flow_compensated_temporal_residual_mae_255'
        lines.append(f"| {clip['id']} | {b['visible_hand_psnr']:.3f} | {g['visible_hand_psnr']:.3f} | {b[key]:.3f} | {g[key]:.3f} |")
    lines+=['','Temporal error is GT-flow-compensated temporal residual MAE inside the common visible hand mask; lower is better, but blur can affect it.',
        'This is a deliberately selected showcase, not the previous frozen 183-crop benchmark. Fixed temporal crops, BF16 and video noise policy differ from that benchmark.',
        'All comparison outputs are raw generations. Full videos paste only the selected edit-mask region back into original frames.',
        'See provenance.json, frames.jsonl, individual report.json files and ../screening/selection.json for exact inputs and settings.','']
    (root/'README.md').write_text('\n'.join(lines))
    print(root/'index.html')


if __name__=='__main__':main()
