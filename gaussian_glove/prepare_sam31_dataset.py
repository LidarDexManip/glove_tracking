"""Quality-filter SAM 3.1 delivery and export training-only Gaussian glove crops."""
from pathlib import Path
import argparse, collections, csv, hashlib, json
import cv2
import numpy as np
import torch
from gaussian_glove.geometry_inputs import (ROOT, camera_matrix, choose_indices,
    subject_betas, vertices_for_session, make_contact_sheet)
from gaussian_glove.quality import DEFAULTS, load_exceptions, exception_reasons, silhouette, evaluate
from gaussian_hand.data.geometry import crop_intrinsics, project_points, square_crop
from gaussian_hand.model.mano import LearnableMANO
from gaussian_hand.model.pose_refinement import axis_angle_to_quaternion

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def preflight(root, source):
    if root.resolve()!=Path(source['snapshot']).resolve():raise ValueError('Use the synchronized HF snapshot, not a local fallback')
    for name,expected in source['file_sha256'].items():
        if sha(root/name)!=expected:raise ValueError('Source changed: '+name)
    with (root/'session_index.csv').open(newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    names=[r['session'] for r in rows if r['group']=='gaussian_hand_display']
    if 'session_040' in names:raise ValueError('Withdrawn session in source index')
    sessions=[root/'gaussian_hand_display'/name for name in names]
    for s in sessions:
        meta=json.loads((s/'metadata.json').read_text());mm=json.loads((s/'mask_metadata.json').read_text())
        if mm['generation']['model']!='SAM 3.1 multiplex':raise ValueError('Expected SAM 3.1: '+s.name)
        if sha(s/'hand_masks.npz')!=mm['sha256']:raise ValueError('Mask checksum mismatch')
        with np.load(s/'mano.npz') as mano,np.load(s/'hand_masks.npz') as masks:
            np.testing.assert_array_equal(mano['timestamps_s'],masks['timestamps_s'])
            if len(mano['timestamps_s'])!=meta['frame_count'] or str(masks['bitorder'])!='little':raise ValueError('Invalid frame contract')
            if np.any(np.diff(mano['timestamps_s'])<=0):raise ValueError('Non-increasing timestamps')
            for side in meta['hands_present']:
                if sha(ROOT/'mano_v1_2/models'/f'MANO_{side.upper()}.pkl')!=meta['mano']['model_sha256'][side]:raise ValueError('MANO model differs')
                if masks[f'{side}_packed'].shape!=(meta['frame_count'],meta['camera']['height'],(meta['camera']['width']+7)//8):raise ValueError('Mask shape mismatch')
                for key in ['hand_pose','global_orient','transl','betas']:
                    if not np.isfinite(mano[f'{side}_{key}']).all():raise ValueError('Nonfinite MANO: '+s.name)
        cap=cv2.VideoCapture(str(s/'rgb.mp4'))
        if not cap.isOpened() or round(cap.get(cv2.CAP_PROP_FRAME_COUNT))!=meta['frame_count']:raise ValueError('RGB count mismatch')
        cap.release()
    return sessions

def prepare_side(args,sessions,side,source,exceptions):
    output=args.output_root/side;output.mkdir(parents=True)
    for folder in ['images','masks','supervision']: (output/folder).mkdir()
    model=LearnableMANO(str(ROOT/'mano_v1_2/models'/f'MANO_{side.upper()}.pkl'),handedness=side,pose_representation='axis_angle')
    faces=model.faces.numpy();betas,deviations=subject_betas(sessions,side)
    records=[];rejected=[];counts=collections.Counter();policy=dict(DEFAULTS)
    for s in sessions:
        meta=json.loads((s/'metadata.json').read_text())
        if side not in meta['hands_present']:continue
        camera=meta['camera'];width,height=camera['width'],camera['height'];K=camera_matrix(meta)
        with np.load(s/'mano.npz') as d,np.load(s/'hand_masks.npz') as archive:
            n=meta['frame_count'];indices=choose_indices(n,args.limit_per_session)
            vertices=vertices_for_session(model,d,side,indices,betas)
            pose=d[f'{side}_hand_pose'][indices];orient=d[f'{side}_global_orient'][indices];trans=d[f'{side}_transl'][indices];times=d['timestamps_s'][indices]
            packed=archive[f'{side}_packed'];other=archive[f'{"right" if side=="left" else "left"}_packed']
            lookup=np.array([int(i).bit_count() for i in range(256)],np.uint8);areas=lookup[packed].sum((1,2))
            cap=cv2.VideoCapture(str(s/'rgb.mp4'));selected={int(f):i for i,f in enumerate(indices)};accepted=0
            for f in range(n):
                if not cap.grab():raise ValueError(f'Cannot decode {s.name}:{f}')
                if f not in selected:continue
                i=selected[f];reasons=exception_reasons(exceptions,s.name,side,f)
                uv,z=project_points(vertices[i],np.eye(4),K)
                visible=(z>.01)&(uv[:,0]>=0)&(uv[:,0]<width)&(uv[:,1]>=0)&(uv[:,1]<height)
                if visible.mean()<policy['min_visible_vertices']:reasons.append('out_of_view')
                box=square_crop(uv[z>.01],width,height,args.crop_scale)
                if box is None:reasons.append('invalid_crop')
                mask=np.unpackbits(packed[f],axis=1,count=width,bitorder='little').astype(bool)
                small=mask[::4,::4];mesh=silhouette(uv/4,z,faces,small.shape[1],small.shape[0])
                positive=z[z>.01];radius=max(1,int(np.ceil(camera['fx']*policy['normal_offset_max_m']/max(float(np.median(positive)) if len(positive) else .01,.01)/4))+1)
                neighbors=[float(areas[f-1]/16),float(areas[f+1]/16)] if 0<f<n-1 else []
                more,metrics,valid=evaluate(small,mesh,radius,policy,neighbors);reasons+=more
                if len(other) and mask.any():
                    overlap=int(lookup[np.bitwise_and(packed[f],other[f])].sum())/int(areas[f])
                    if overlap>.05:reasons.append('cross_hand_overlap')
                if reasons:
                    rejected.append({'session':s.name,'hand':side,'frame':f,'reasons':sorted(set(reasons)),'metrics':metrics});counts.update(set(reasons));continue
                ok,image=cap.retrieve()
                if not ok:raise ValueError(f'Cannot read {s.name}:{f}')
                x0,y0,x1,y1=box;size=(args.resolution,args.resolution)
                crop=cv2.resize(image[y0:y1,x0:x1],size,interpolation=cv2.INTER_AREA)
                cropmask=cv2.resize(mask[y0:y1,x0:x1].astype(np.uint8)*255,size,interpolation=cv2.INTER_NEAREST)
                supervision=cv2.resize(valid.astype(np.uint8)*255,(width,height),interpolation=cv2.INTER_NEAREST)
                supervision=cv2.resize(supervision[y0:y1,x0:x1],size,interpolation=cv2.INTER_NEAREST)
                stem=f'{s.name}_{f:06d}';paths={'image':f'images/{stem}.jpg','mask':f'masks/{stem}.png','supervision':f'supervision/{stem}.png'}
                for key,data in [('image',crop),('mask',cropmask),('supervision',supervision)]:
                    if not cv2.imwrite(str(output/paths[key]),data,[cv2.IMWRITE_JPEG_QUALITY,95] if key=='image' else []):raise IOError(paths[key])
                records.append({'sequence':f'black_glove_{s.name}','frame_index':f,'timestamp_s':float(times[i]),'handedness':side,'split':'train',**paths,'mask_source':'hf_sam3.1','quality_pass':True,'quality_metrics':metrics,'source_revision':source['revision'],'output_size':args.resolution,'crop_xyxy':list(map(int,box)),'crop_intrinsics':crop_intrinsics(K,box,args.resolution).tolist(),'world_to_camera':np.eye(4).tolist(),'mano_pca':pose[i].tolist(),'pose_representation':'axis_angle','wrist_q_wxyz':axis_angle_to_quaternion(torch.from_numpy(orient[i])).tolist(),'wrist_t_world':trans[i].tolist(),'mano_betas':betas.tolist(),'visible_mano_ratio':float(visible.mean()),'source_video':str(s/'rgb.mp4')});accepted+=1
            cap.release()
        print(json.dumps({'session':s.name,'side':side,'accepted':accepted}),flush=True)
    if not records:raise ValueError('No qualified frames: '+side)
    with (output/'frames.jsonl').open('w') as f:
        for r in records:f.write(json.dumps(r,separators=(',',':'))+'\n')
    with (output/'rejected_frames.jsonl').open('w') as f:
        for r in rejected:f.write(json.dumps(r,separators=(',',':'))+'\n')
    summary={'format_version':2,'frames':len(records),'rejected':len(rejected),'rejection_reasons':dict(counts),'source_revision':source['revision'],'quality_policy':policy,'training_split':'all','mask_source':'hf_sam3.1','offset_max_m':policy['normal_offset_max_m'],'subject_betas':betas.tolist(),'session_beta_max_abs_deviation':deviations,'cuff_policy':'Ignore glove pixels outside bounded MANO neighborhood; do not clip the renderer.','frames_sha256':sha(output/'frames.jsonl')}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');make_contact_sheet(records,output,output/'contact_sheet.jpg',96)
    return summary

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw-root',type=Path,default=ROOT/'data/ego-clinical-task');p.add_argument('--output-root',type=Path,default=ROOT/'gaussian_glove/data/processed_sam31_offset');p.add_argument('--sides',nargs='+',choices=['left','right'],default=['left','right']);p.add_argument('--resolution',type=int,default=512);p.add_argument('--crop-scale',type=float,default=1.35);p.add_argument('--limit-per-session',type=int,default=0);p.add_argument('--threads',type=int,default=4);args=p.parse_args()
    if args.output_root.exists():raise FileExistsError('Use a fresh output directory: '+str(args.output_root))
    cv2.setNumThreads(1);torch.set_num_threads(args.threads)
    source=json.loads((ROOT/'gaussian_glove/SOURCE.json').read_text());sessions=preflight(args.raw_root,source);exceptions=load_exceptions(args.raw_root/'quality_exceptions.csv')
    summaries={side:prepare_side(args,sessions,side,source,exceptions) for side in args.sides}
    (args.output_root/'preparation.json').write_text(json.dumps({'source_revision':source['revision'],'complete':True,'sides':summaries},indent=2))
    print(json.dumps(summaries),flush=True)

if __name__=='__main__':main()
