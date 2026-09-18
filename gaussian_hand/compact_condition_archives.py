"""Preserve regression images, then explicitly retire full visual archives.

--prepare only copies and verifies; --delete-full requires preparation and a
live retained-sample audit. Source/render/model assets are never deleted.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
from pathlib import Path
import tarfile

from gaussian_hand.archive_retention import (
    REGRESSION_CLIPS, RETENTION_FILE, sha256, validate_archive_storage,
)
from gaussian_hand.prepare_diffusion_strict import OUT, ROOT, CONFIG


def selected_records(records, seen):
    chosen = {0, len(records) // 2, len(records) - 1}
    for i, record in enumerate(records):
        key = (record['sequence_id'], record['frame_index'], record.get('adaptive_handedness'))
        if key in seen:
            chosen.add(i)
        for seq, start, side in REGRESSION_CLIPS:
            if key[0] == seq and key[2] == side and start <= key[1] < start + 150:
                chosen.add(i)
    return [records[i] for i in sorted(chosen)]


def stat_signature(path):
    s = path.stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns]


def prepare(root):
    marker = root / RETENTION_FILE
    if marker.exists():
        raise FileExistsError(marker)
    ready = json.loads((root / 'READY.json').read_text())
    historical = json.loads((root / 'dataset_audit.json').read_text())
    if ready['dataset_audit_sha256'] != sha256(root / 'dataset_audit.json') or historical['status'] != 'passed':
        raise ValueError('Historical audit binding failed')
    audit_rows = {r['sequence']: r for r in historical['rows']}
    config = json.loads(CONFIG.read_text())
    seen = {(r['sequence_id'], r['frame_index'], r.get('adaptive_handedness'))
            for r in map(json.loads, (ROOT / config['data']['val_manifest']).read_text().splitlines())}
    result = {'schema_version': 1, 'policy': 'regression_samples',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'bindings': {n: sha256(root / n) for n in ('READY.json', 'dataset_audit.json')},
              'selection': 'All five 150-frame demo clips, seen-eval matches, first/middle/last per sequence',
              'clips': REGRESSION_CLIPS, 'sequences': {}}
    for row in ready['rows']:
        d = root / 'image_sequences' / row['sequence']
        if not (d / 'images.tar').is_file() or (d / 'images.tar').is_symlink():
            raise FileNotFoundError(d / 'images.tar')
        if (d / 'images.tar').stat().st_size != row['archive_bytes']:
            raise ValueError(f'Wrong original size: {d}')
        if sha256(d / 'index.jsonl') != audit_rows[row['sequence']]['index_sha256']:
            raise ValueError(f'Historical index changed: {d}')
        if sha256(d / '_SUCCESS.json') != row['images_success_sha256']:
            raise ValueError(f'Historical completion marker changed: {d}')
        for name in ('samples.tar', 'samples.partial.tar', 'samples.jsonl'):
            if (d / name).exists():
                raise FileExistsError(d / name)
    for row in ready['rows']:
        seq = row['sequence']; directory = root / 'image_sequences' / seq
        records = [json.loads(line) for line in (directory / 'index.jsonl').read_text().splitlines()]
        selected = selected_records(records, seen)
        expected = {r[k] for r in selected for k in ('condition_member', 'overlay_member')}
        for clip_seq, start, side in REGRESSION_CLIPS:
            if seq == clip_seq:
                frames = {r['frame_index'] for r in selected if r.get('adaptive_handedness') == side}
                if not set(range(start, start + 150)) <= frames:
                    raise ValueError(f'Incomplete regression clip: {seq}')
        member_hashes = {}
        with tarfile.open(directory / 'images.tar', 'r:') as original, \
                tarfile.open(directory / 'samples.partial.tar', 'x:') as output:
            for member in original:
                if member.name not in expected:
                    continue
                if not member.isfile() or member.name in member_hashes:
                    raise ValueError(f'Invalid archive member: {member.name}')
                content = original.extractfile(member).read()
                member_hashes[member.name] = hashlib.sha256(content).hexdigest()
                output.addfile(member, io.BytesIO(content))
        if set(member_hashes) != expected:
            raise ValueError(f'Missing selected images: {seq}')
        with tarfile.open(directory / 'samples.partial.tar', 'r:') as retained:
            actual = {m.name: hashlib.sha256(retained.extractfile(m).read()).hexdigest() for m in retained}
        if actual != member_hashes:
            raise ValueError(f'Retained content mismatch: {seq}')
        (directory / 'samples.partial.tar').rename(directory / 'samples.tar')
        (directory / 'samples.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in selected))
        result['sequences'][seq] = {
            'original_bytes': row['archive_bytes'], 'original_stat': stat_signature(directory / 'images.tar'),
            'record_indices': [r['global_record_index'] for r in selected], 'member_sha256': member_hashes,
            'sha256': {n: sha256(directory / n) for n in ('samples.tar', 'samples.jsonl', 'index.jsonl', '_SUCCESS.json')},
        }
        print(f'PRESERVED {seq}: {len(selected)} records', flush=True)
    marker.write_text(json.dumps(result, indent=2) + '\n')
    validate_archive_storage(root)
    print('Prepared and verified. Full archives have NOT been deleted.')


def delete_full(root):
    retention = validate_archive_storage(root)
    if retention is None:
        raise ValueError('Prepare regression samples first')
    audit = json.loads((root / 'retained_sample_audit.json').read_text())
    if audit['status'] != 'passed' or audit['retention_sha256'] != sha256(root / RETENTION_FILE):
        raise ValueError('Live retained-sample audit missing or stale')
    with ExitStack() as stack:
        targets = []
        for seq, entry in retention['sequences'].items():
            directory = root / 'image_sequences' / seq
            if Path(seq).name != seq or directory.is_symlink():
                raise ValueError('Unexpected deletion target')
            lock = stack.enter_context((directory / '.export.lock').open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            target = directory / 'images.tar'
            if target.exists():
                if target.is_symlink() or stat_signature(target) != entry['original_stat']:
                    raise ValueError(f'Original changed since preparation: {target}')
                targets.append(target)
        size = sum(p.stat().st_size for p in targets)
        for target in targets:
            target.unlink()
            print(f'DELETED {target}', flush=True)
    validate_archive_storage(root)
    (root / 'ARCHIVE_CLEANUP.json').write_text(json.dumps({
        'completed_utc': datetime.now(timezone.utc).isoformat(), 'deleted_archives': len(targets),
        'deleted_bytes': size, 'retention_sha256': sha256(root / RETENTION_FILE),
        'recovery': 'Re-export preserved source/render/mask data into a separate output directory; no full archive backup',
    }, indent=2) + '\n')
    print(f'Complete: removed {len(targets)} archives, {size / 1024**3:.2f} GiB')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--delete-full', action='store_true')
    args = parser.parse_args()
    root = OUT.resolve()
    if root != Path('/data/szeng87/glove_tracking_data/outputs/gaussian_hand/diffusion_strict_v2'):
        raise ValueError('This maintenance command is scoped to the audited local strict dataset')
    (prepare if args.prepare else delete_full)(root)


if __name__ == '__main__':
    main()
