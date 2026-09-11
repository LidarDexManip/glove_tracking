"""CPU-only final archive and real training-dataloader checks. Never train."""
from __future__ import annotations

import argparse
import json
import tarfile
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from gaussian_hand.prepare_diffusion_strict import CONFIG, OUT, ROOT, digest
from train_hand_restorer import make_hugg_aria_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-for-preparation', action='store_true')
    args = parser.parse_args()
    if args.wait_for_preparation:
        while True:
            state = json.loads((OUT / 'preparation_status.json').read_text())
            if state['status'] == 'failed':
                raise RuntimeError(f"Preparation failed: {state.get('error')}")
            if state['status'] == 'ready':
                break
            time.sleep(30)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    config = json.loads(CONFIG.read_text())
    rows = []
    first_by_subject = {}
    for manifest in sorted((OUT / 'manifests').glob('*.jsonl')):
        sequence = manifest.stem
        records = [json.loads(line) for line in manifest.open()]
        path = OUT / 'image_sequences' / sequence
        success = json.loads((path / '_SUCCESS.json').read_text())
        assert success['status'] == 'complete'
        assert success['config_sha256'] == digest(CONFIG)
        assert success['manifest_sha256'] == digest(manifest)
        assert success['render_success_sha256'] == digest(OUT / 'aligned' / sequence / '_SUCCESS.json')
        assert success['outside_edit_max_delta'] == 0
        assert success['records'] == len(records)
        assert success['archive_bytes'] == (path / 'images.tar').stat().st_size
        index = [json.loads(line) for line in (path / 'index.jsonl').open()]
        assert len(index) == len(records)
        expected = []
        for record, entry in zip(records, index):
            assert all(entry[key] == value for key, value in record.items())
            stem = f"{record['global_record_index']:07d}"
            assert entry['condition_member'] == f'{stem}.condition.png'
            assert entry['overlay_member'] == f'{stem}.overlay.jpg'
            expected.extend([entry['condition_member'], entry['overlay_member']])
        with tarfile.open(path / 'images.tar') as archive:
            members = archive.getmembers()
            assert [m.name for m in members] == expected
            assert all(m.isfile() and m.size > 0 for m in members)
        first_by_subject.setdefault(sequence.split('_')[0], records[0]['global_record_index'])
        rows.append({'sequence': sequence, 'records': len(records), 'members': len(expected),
                     'index_sha256': digest(path / 'index.jsonl')})
        print(json.dumps({'archive_verified': sequence, 'records': len(records)}), flush=True)
    assert len(rows) == 136 and sum(r['records'] for r in rows) == 524917
    assert len(first_by_subject) == 9

    # Use the actual training dataset factory/default collator, without numpy
    # diagnostic fields or any model, optimizer, Accelerator, or CUDA setup.
    dataset = make_hugg_aria_dataset(config, ROOT / config['data']['train_manifest'])
    indices = sorted(first_by_subject.values())
    loader = DataLoader(Subset(dataset, indices), batch_size=3, num_workers=0)
    batches = []
    for batch in loader:
        shapes = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                assert torch.isfinite(value).all(), key
                shapes[key] = list(value.shape)
        batches.append(shapes)
    assert len(batches) == 3

    # Decode one lossless PNG per subject and compare with online conditioning.
    visual_dataset = make_hugg_aria_dataset(config, ROOT / config['data']['train_manifest'], include_numpy=True)
    for index in indices:
        sample = visual_dataset[index]
        sequence = visual_dataset.samples[index]['sequence_id']
        with tarfile.open(OUT / 'image_sequences' / sequence / 'images.tar') as archive:
            content = archive.extractfile(f'{index:07d}.condition.png').read()
        decoded = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
        expected = (sample['condition_rgb_np'][..., ::-1] * 255).round().clip(0, 255).astype(np.uint8)
        assert np.array_equal(decoded, expected), sequence
    report = {'status': 'passed', 'training_started': False, 'config_sha256': digest(CONFIG),
              'sequences': len(rows), 'records': sum(r['records'] for r in rows),
              'image_members': sum(r['members'] for r in rows), 'archive_headers_and_all_indices_verified': True,
              'live_dataloader_subjects': sorted(first_by_subject), 'live_dataloader_batches': batches,
              'archive_png_vs_live_samples': indices, 'archive_png_vs_live_max_delta': 0, 'rows': rows}
    report_path = OUT / 'dataset_audit.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    ready_path = OUT / 'READY.json'
    ready = json.loads(ready_path.read_text())
    ready['dataset_audit_sha256'] = digest(report_path)
    ready['asset_lock_sha256'] = digest(OUT / 'ASSET_LOCK.json')
    temporary = OUT / 'READY.audited.tmp'
    temporary.write_text(json.dumps(ready, indent=2) + '\n')
    temporary.replace(ready_path)
    from gaussian_hand.diffusion_readiness import validate_ready
    validate_ready(CONFIG)
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}), flush=True)


if __name__ == '__main__':
    main()
