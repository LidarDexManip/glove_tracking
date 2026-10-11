"""Export all task frames using pinned Gaussian gloves; no quality filtering.

Run from the server repository. Four workers own disjoint complete sessions.
Native render RGB and alpha are lossless uint8 FFV1; overlays are previews.
Training consumes lossless PNG crops in per-session tar files, never MP4.
"""
from __future__ import annotations
import argparse, hashlib, io, json, os, subprocess, tarfile, time
from pathlib import Path
import cv2
import numpy as np
import torch
from gsplat import rasterization_2dgs
from gaussian_glove.render_overlay_video import load_renderer, camera_matrix
from gaussian_hand.data.geometry import project_points, rasterize_silhouette
from gaussian_hand.model.pose_refinement import axis_angle_to_quaternion
from hand_restoration.adaptive_hand_crops import adaptive_hand_crop_instances
from hand_restoration.hugg_aria_conditioning import CropBox, build_faithful_condition, crop_resize

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()

def write_json(path, value):
    p = Path(path); tmp = p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n'); tmp.replace(p)

def boundary(faces):
    edges = np.sort(np.concatenate([faces[:,[0,1]], faces[:,[1,2]], faces[:,[2,0]]]), axis=1)
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    ids = np.unique(edges[counts == 1]); assert len(ids) == 16, ids
    return ids

def encoder(path, w, h, fps, gray=False, preview=False):
    args = ['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-n','-threads','1',
            '-f','rawvideo','-pix_fmt','gray' if gray else 'rgb24','-s',f'{w}x{h}',
            '-r',str(fps),'-i','pipe:0','-an']
    args += (['-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p',
              '-threads','1','-movflags','+faststart'] if preview else
             ['-c:v','ffv1','-level','3','-threads','1','-pix_fmt','gray' if gray else 'bgr0'])
    return subprocess.Popen(args+[str(path)], stdin=subprocess.PIPE)

def u8(x): return np.rint(np.clip(x,0,1)*255).astype(np.uint8)

def add_png(archive, name, value):
    x = np.asarray(value)
    if x.dtype != np.uint8: x = u8(x)
    if x.ndim == 3: x = x[...,::-1]
    ok, data = cv2.imencode('.png', x, [cv2.IMWRITE_PNG_COMPRESSION, 2])
    if not ok: raise RuntimeError(name)
    payload = data.tobytes(); entry = tarfile.TarInfo(name); entry.size = len(payload)
    archive.addfile(entry, io.BytesIO(payload))

def montage(images):
    panels=[]
    for title, x in images:
        x = u8(x) if x.dtype != np.uint8 else x.copy()
        if x.ndim==2: x=np.repeat(x[...,None],3,axis=2)
        x=cv2.resize(x,(256,256),interpolation=cv2.INTER_AREA)
        cv2.rectangle(x,(0,0),(256,23),(0,0,0),-1)
        cv2.putText(x,title,(5,16),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1)
        panels.append(x)
    return np.concatenate(panels,axis=1)

@torch.inference_mode()
def process(row, models, inventory, args):
    sequence=row['sequence_id']; source=Path(row['path']); out=args.output/sequence
    success=out/'_SUCCESS.json'
    if success.exists():
        old=json.loads(success.read_text())
        if old['inventory_sha256']!=digest(args.inventory) or old['script_sha256']!=digest(__file__):
            raise RuntimeError('Completed session belongs to different input/code: '+sequence)
        return old
    out.mkdir(parents=True,exist_ok=True)
    if any((out/name).exists() for name in ['render.mkv','alpha.mkv','overlay.mp4','crops.tar']):
        raise FileExistsError('Incomplete outputs exist; use a fresh output root: '+str(out))
    meta=json.loads((source/'metadata.json').read_text()); cam=meta['camera']
    w,h=int(cam['width']),int(cam['height']); K=camera_matrix(meta); fps=float(cam['fps_nominal'])
    n=int(meta['frame_count']); assert n==row['frames']
    n=min(n,args.limit_frames) if args.limit_frames else n
    pose=np.load(source/'mano.npz'); masks=np.load(source/'hand_masks.npz')
    np.testing.assert_array_equal(pose['timestamps_s'],masks['timestamps_s'])
    cap=cv2.VideoCapture(str(source/'rgb.mp4')); assert cap.isOpened()
    encs=[encoder(out/'render.mkv',w,h,fps),encoder(out/'alpha.mkv',w,h,fps,gray=True),
          encoder(out/'overlay.mp4',w,h,fps,preview=True)]
    records=[]; preview=[]; start=time.time(); zero_mano=zero_sam=0
    stats={'outside_sam_overlay_max_delta':0.0,'outside_edit_condition_max_delta':0.0}
    ring_ids={s:boundary(models[s].model.mano.faces.cpu().numpy()) for s in models}
    Kt=torch.tensor(K,dtype=torch.float32,device='cuda')[None]
    eye=torch.eye(4,device='cuda')[None]
    try:
        with tarfile.open(out/'crops.tar','w') as tar:
            for f in range(n):
                ok,bgr=cap.read()
                if not ok: raise RuntimeError(f'{sequence}: RGB decode failed at {f}')
                target=bgr[...,::-1].astype(np.float32)/255
                attributes=[]; mano=np.zeros((h,w),bool)
                rings=np.full((2,16,2),np.nan,np.float32); palms=np.full((2,2),np.nan,np.float32)
                sam=np.zeros((h,w),bool)
                for i,side in enumerate(('left','right')):
                    if side not in meta['hands_present']: continue
                    p=torch.tensor(pose[side+'_hand_pose'][f],device='cuda',dtype=torch.float32)
                    q=axis_angle_to_quaternion(torch.tensor(pose[side+'_global_orient'][f],device='cuda',dtype=torch.float32))
                    t=torch.tensor(pose[side+'_transl'][f],device='cuda',dtype=torch.float32)
                    a=models[side].model.attributes(p,q,t)
                    if not all(torch.isfinite(a[k]).all() for k in ('means','quats','scales','opacities','colors')):
                        raise ValueError(f'Nonfinite render input: {sequence}/{f}/{side}')
                    attributes.append(a)
                    vertices=a['vertices']; uv,z=project_points(vertices.cpu().numpy(),np.eye(4),K)
                    mano |= rasterize_silhouette(uv,z,models[side].model.mano.faces.cpu().numpy(),w,h,1).astype(bool)
                    ids=ring_ids[side]
                    if np.all(z[ids]>.01): rings[i]=uv[ids]
                    # MANO proximal finger joints provide a stable palm-axis anchor.
                    joints=models[side].model.mano.layer.J_regressor @ vertices
                    puv,pz=project_points(joints[[1,4,7,10]].mean(0,keepdim=True).cpu().numpy(),np.eye(4),K)
                    if pz[0]>.01: palms[i]=puv[0]
                    packed=masks[side+'_packed'][f]
                    sam |= np.unpackbits(packed,axis=1,count=w,bitorder=str(masks['bitorder'])).astype(bool)
                joined={k:torch.cat([a[k] for a in attributes],0) for k in ('means','quats','scales','opacities','colors')}
                rgb,alpha,*_=rasterization_2dgs(**{k:joined[k] for k in ('means','quats','scales','opacities')},
                    colors=joined['colors'][None],viewmats=eye,Ks=Kt,width=w,height=h,
                    render_mode='RGB',packed=False,distloss=False)
                render=rgb[0,...,:3].clamp(0,1).cpu().numpy(); alpha=alpha[0,...,0].clamp(0,1).cpu().numpy()
                if not np.isfinite(render).all() or not np.isfinite(alpha).all(): raise ValueError('Nonfinite output')
                # Overlay only visible SAM pixels: no object depth is available.
                overlay=target*(1-alpha[...,None]*sam[...,None])+render*sam[...,None]
                stats['outside_sam_overlay_max_delta']=max(stats['outside_sam_overlay_max_delta'],float(np.max(np.abs(overlay[~sam]-target[~sam]),initial=0)))
                for enc,value in zip(encs,(u8(render),u8(alpha),u8(overlay))): enc.stdin.write(value.tobytes())
                zero_mano+=int(not mano.any()); zero_sam+=int(not sam.any())
                if mano.any():
                    instances=adaptive_hand_crop_instances(mano,rings,palms,min_component_area=1)
                    boxes=[(r.box,r.mode,r.handedness) for r in instances]
                else:
                    # Preserve offscreen/empty frames instead of quality-filtering them.
                    boxes=[(CropBox(0,0,w,h),'empty_full_frame','both')]
                for c,(box,mode,hand) in enumerate(boxes):
                    def crop(x,mask=False): return crop_resize(x.astype(np.float32),box,512,cv2.INTER_NEAREST if mask else cv2.INTER_AREA)
                    ct,cr,ca,cm,cs=crop(target),crop(render),crop(alpha),crop(mano,True)>0,crop(sam,True)>0
                    rr=rings.copy(); pp=palms.copy()
                    for pts in (rr,pp):
                        pts[...,0]=(pts[...,0]-box.x0+.5)*512/box.width-.5
                        pts[...,1]=(pts[...,1]-box.y0+.5)*512/box.height-.5
                    faithful=build_faithful_condition(ct,cr,ca,cm,visible_hand_mask=cs,
                        wrist_ring_points=rr,wrist_palm_points=pp,dilation_px=3,
                        wrist_sleeve_orientation='palm_axis',fill_value=0,
                        render_rgb_semantics='premultiplied')
                    delta=float(np.max(np.abs(faithful.condition_rgb[~faithful.edit_mask]-ct[~faithful.edit_mask]),initial=0))
                    stats['outside_edit_condition_max_delta']=max(stats['outside_edit_condition_max_delta'],delta)
                    if delta>1e-6: raise AssertionError('Condition changed outside edit mask')
                    key=f'{f:06d}_{c}'
                    values={'target':ct,'render':cr,'alpha':ca,'condition':faithful.condition_rgb,
                            'overlay':faithful.overlay_rgb,'mano_mask':cm,'visible_mask':faithful.visible_hand_mask,
                            'edit_mask':faithful.edit_mask,'sam_mask':cs}
                    for name,value in values.items(): add_png(tar,f'{key}.{name}.png',value)
                    records.append({'key':key,'sequence_id':sequence,'frame_index':f,'frame_id':f'{f:06d}',
                        'timestamp_s':float(pose['timestamps_s'][f]),'crop_xyxy':box.as_list(),'crop_mode':mode,
                        'handedness':hand,'shard':f'{sequence}/crops.tar','output_size':512,
                        'camera_id':'zed_rectified_left_camera','source_revision':inventory['source_revision']})
                    if f in set(np.linspace(0,n-1,4,dtype=int)):
                        preview.append(montage([('TARGET '+key,ct),('GAUSSIAN',cr),('SAM OVERLAY',faithful.overlay_rgb),
                                                ('CONDITION',faithful.condition_rgb),('EDIT MASK',faithful.edit_mask)]))
                if (f+1)%50==0 or f==n-1:
                    progress={'sequence':sequence,'frames':f+1,'total':n,'crops':len(records),'seconds':time.time()-start}
                    write_json(out/'progress.json',progress); print(json.dumps(progress),flush=True)
    finally:
        cap.release(); pose.close(); masks.close()
        for enc in encs:
            if enc.stdin: enc.stdin.close()
        codes=[enc.wait() for enc in encs]
    if any(codes): raise RuntimeError(f'ffmpeg errors: {codes}')
    (out/'frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    np.save(out/'timestamps_s.npy',np.load(source/'mano.npz')['timestamps_s'][:n])
    if preview: cv2.imwrite(str(out/'contact_sheet.jpg'),np.concatenate(preview)[...,::-1])
    result={'sequence_id':sequence,'frames':n,'crops':len(records),'source_frames':row['frames'],
            'native_size':[w,h],'fps_nominal':fps,'elapsed_seconds':time.time()-start,
            'empty_mano_frames':zero_mano,'empty_sam_frames':zero_sam,**stats,
            'inventory_sha256':digest(args.inventory),'script_sha256':digest(__file__),
            'checkpoint_sha256':{s:v['sha256'] for s,v in inventory['checkpoints'].items()},
            'semantics':'joint depth-sorted 2DGS; premultiplied RGB; native alpha; SAM-visible overlay; no quality filtering',
            'crop_condition_policy':'bare-hand adaptive crops and faithful MANO/SAM/wrist conditioning; uint8 PNG cache',
            'artifacts':{p.name:{'bytes':p.stat().st_size,'sha256':digest(p)} for p in out.iterdir() if p.is_file() and p.name!='progress.json'}}
    write_json(success,result); return result

def main():
    p=argparse.ArgumentParser(); p.add_argument('--inventory',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--worker',type=int,default=0)
    p.add_argument('--workers',type=int,default=1); p.add_argument('--limit-frames',type=int,default=0)
    p.add_argument('--limit-sessions',type=int,default=0); args=p.parse_args()
    cv2.setNumThreads(1); torch.set_num_threads(1)
    inventory=json.loads(args.inventory.read_text()); assert len(inventory['sessions'])==14
    models={}
    for side,item in inventory['checkpoints'].items():
        assert digest(item['path'])==item['sha256']; models[side]=load_renderer(Path(item['path']),side,'cuda')
    rows=inventory['sessions'][args.worker::args.workers]
    if args.limit_sessions: rows=rows[:args.limit_sessions]
    for row in rows: process(row,models,inventory,args)

if __name__=='__main__': main()
