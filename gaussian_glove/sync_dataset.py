"""Pin and verify the Hugging Face delivery used by Gaussian glove training."""
from pathlib import Path
import argparse, hashlib, json, os
from huggingface_hub import HfApi, snapshot_download
ROOT = Path(__file__).resolve().parents[1]
REPO = 'LIDAR-GT/ego-clinical-task'

def sync(revision=None):
    api = HfApi()
    info = api.dataset_info(REPO, revision=revision, files_metadata=True)
    revision = info.sha
    snapshot = Path(snapshot_download(REPO, repo_type='dataset', revision=revision,
                                     cache_dir=str(ROOT / 'gaussian_glove/data/hf_cache'), max_workers=4))
    files = {}
    for entry in info.siblings:
        path = snapshot / entry.rfilename
        payload = path.read_bytes()
        sha = hashlib.sha256(payload).hexdigest()
        if entry.lfs:
            if sha != entry.lfs.sha256: raise ValueError('HF checksum mismatch: ' + entry.rfilename)
        elif hashlib.sha1(b'blob ' + str(len(payload)).encode() + b'\0' + payload).hexdigest() != entry.blob_id:
            raise ValueError('HF git blob mismatch: ' + entry.rfilename)
        files[entry.rfilename] = sha
    if (snapshot / 'gaussian_hand_display/session_040').exists():
        raise ValueError('Withdrawn session_040 cannot be used')
    link = ROOT / 'data/ego-clinical-task'
    if link.exists() and not link.is_symlink():
        raise ValueError('Refusing to replace a non-symlink dataset directory: ' + str(link))
    temporary = link.with_name('.ego-clinical-task-sync')
    if temporary.is_symlink(): temporary.unlink()
    temporary.symlink_to(snapshot, target_is_directory=True)
    os.replace(temporary, link)
    source = {'repository': REPO, 'revision': revision, 'snapshot': str(snapshot),
              'file_sha256': files, 'policy': 'Training reads this verified HF snapshot; no local mocap fallback.'}
    (ROOT / 'gaussian_glove/SOURCE.json').write_text(json.dumps(source, indent=2) + '\n')
    print(json.dumps({'revision': revision, 'files': len(files), 'dataset_root': str(link)}), flush=True)
    return source

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--revision'); a = p.parse_args(); sync(a.revision)
