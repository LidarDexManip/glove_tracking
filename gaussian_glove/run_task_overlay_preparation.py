"""Supervise four authorized overlay workers and verify complete frame coverage."""
import argparse, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
from gaussian_glove.prepare_task_overlays import digest, write_json

def main():
    p=argparse.ArgumentParser(); p.add_argument('--inventory',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'READY.json').exists(): raise FileExistsError('Already prepared')
    inventory=json.loads(a.inventory.read_text()); workers=[]; logs=[]; start=time.time()
    write_json(a.output/'provenance.json',{'inventory':inventory,'inventory_sha256':digest(a.inventory),
        'script_sha256':digest(Path(__file__).with_name('prepare_task_overlays.py')),
        'gpus':[0,1,2,3],'scope':'all 14 task sessions; no quality filtering; no diffusion launch'})
    for gpu in range(4):
        log=(a.output/f'worker_{gpu}.log').open('w'); logs.append(log)
        env=os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        cmd=[sys.executable,'-m','gaussian_glove.prepare_task_overlays','--inventory',str(a.inventory),
             '--output',str(a.output),'--worker',str(gpu),'--workers','4']
        workers.append(subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env))
    try:
        while True:
            codes=[x.poll() for x in workers]
            progress=[]
            for row in inventory['sessions']:
                path=a.output/row['sequence_id']/'progress.json'
                if path.exists(): progress.append(json.loads(path.read_text()))
            state={'state':'rendering','elapsed_seconds':time.time()-start,'pids':[x.pid for x in workers],
                   'returncodes':codes,'completed_sessions':len(list(a.output.glob('*/*/_SUCCESS.json'))),
                   'rendered_frames':sum(x['frames'] for x in progress),'expected_frames':6300,'sessions':progress}
            write_json(a.output/'status.json',state)
            if any(c is not None and c!=0 for c in codes): raise RuntimeError(f'Worker failed: {codes}')
            if all(c is not None for c in codes): break
            time.sleep(15)
        all_records=[]; seen=[]; summaries=[]
        for row in inventory['sessions']:
            folder=a.output/row['sequence_id']; report=json.loads((folder/'_SUCCESS.json').read_text())
            assert report['frames']==row['frames']
            records=[json.loads(x) for x in (folder/'frames.jsonl').read_text().splitlines()]
            assert sorted(set(x['frame_index'] for x in records))==list(range(row['frames']))
            assert len({x['key'] for x in records})==len(records)
            times=np.load(folder/'timestamps_s.npy')
            with np.load(Path(row['path'])/'mano.npz') as src: np.testing.assert_array_equal(times,src['timestamps_s'])
            for name in ('render.mkv','alpha.mkv','overlay.mp4'):
                raw=subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
                    '-show_entries','stream=width,height,nb_read_frames','-of','json',str(folder/name)],text=True)
                stream=json.loads(raw)['streams'][0]
                assert int(stream['nb_read_frames'])==row['frames'], (name,stream)
                assert [stream['width'],stream['height']]==report['native_size']
            all_records.extend(records)
            wanted=set(np.linspace(0,row['frames']-1,8,dtype=int))
            seen.extend(x for x in records if x['frame_index'] in wanted)
            summaries.append({k:report[k] for k in ('sequence_id','frames','crops','empty_mano_frames','empty_sam_frames')})
        assert sum(x['frames'] for x in summaries)==6300
        for name,rows in [('train_manifest.jsonl',all_records),('seen_eval_manifest.jsonl',seen)]:
            (a.output/name).write_text(''.join(json.dumps(x)+'\n' for x in rows))
        result={'state':'ready','sessions':summaries,'source_frames':6300,'training_crops':len(all_records),
                'seen_eval_crops':len(seen),'all_sessions_train':True,'elapsed_seconds':time.time()-start,
                'evaluation_policy':'seen training reconstruction; no held-out evaluation',
                'train_manifest_sha256':digest(a.output/'train_manifest.jsonl'),
                'seen_eval_manifest_sha256':digest(a.output/'seen_eval_manifest.jsonl')}
        write_json(a.output/'READY.json',result); write_json(a.output/'status.json',result)
        print(json.dumps(result),flush=True)
    except BaseException as e:
        for worker in workers:
            if worker.poll() is None: worker.terminate()
        write_json(a.output/'status.json',{'state':'failed','error':repr(e),'elapsed_seconds':time.time()-start})
        raise
    finally:
        for log in logs: log.close()

if __name__=='__main__': main()
