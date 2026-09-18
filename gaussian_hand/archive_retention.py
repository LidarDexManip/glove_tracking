"""Validate compacted visual archives without relaxing training-data checks."""
import hashlib
import json
from pathlib import Path

RETENTION_FILE = 'ARCHIVE_RETENTION.json'
REGRESSION_CLIPS = [
    ('P0001_f6cc0cc8', 1018, 'right'),
    ('P0002_65085bfc', 2553, 'left'),
    ('P0009_0e02bf39', 1453, 'both'),
    ('P0012_0323d770', 2262, 'both'),
    ('P0014_24cb3bf0', 997, 'both'),
]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_archive_storage(root, require_full=False):
    """Missing full archives require a valid, hash-bound retention manifest."""
    root = Path(root)
    ready = json.loads((root / 'READY.json').read_text())
    marker = root / RETENTION_FILE
    retention = json.loads(marker.read_text()) if marker.exists() else None
    if retention is not None:
        if retention.get('schema_version') != 1 or retention.get('policy') != 'regression_samples':
            raise ValueError('Unknown archive retention policy')
        for name in ('READY.json', 'dataset_audit.json'):
            if retention['bindings'].get(name) != sha256(root / name):
                raise ValueError(f'Archive retention binding changed: {name}')
        if set(retention['sequences']) != {r['sequence'] for r in ready['rows']}:
            raise ValueError('Incomplete archive retention coverage')
    for row in ready['rows']:
        seq = row['sequence']
        directory = root / 'image_sequences' / seq
        if retention is not None:
            entry = retention['sequences'][seq]
            if entry['original_bytes'] != row['archive_bytes']:
                raise ValueError(f'Original archive size mismatch: {seq}')
            if not entry['record_indices'] or len(set(entry['record_indices'])) != len(entry['record_indices']):
                raise ValueError(f'Invalid retained record selection: {seq}')
            for name in ('samples.tar', 'samples.jsonl', 'index.jsonl', '_SUCCESS.json'):
                if sha256(directory / name) != entry['sha256'][name]:
                    raise ValueError(f'Retained archive artifact changed: {seq}/{name}')
        full = directory / 'images.tar'
        if full.is_file():
            if full.stat().st_size != row['archive_bytes']:
                raise ValueError(f'Full archive size mismatch: {seq}')
        elif require_full or retention is None:
            raise FileNotFoundError(f'{full}: full visual audit needs original archives; '
                                    'use --retained after intentional compaction')
    return retention


def regression_archive(root, sequence):
    root = Path(root)
    full = root / 'image_sequences' / sequence / 'images.tar'
    if full.is_file():
        return full
    retention = validate_archive_storage(root)
    if retention is None or sequence not in retention['sequences']:
        raise FileNotFoundError(full)
    return full.with_name('samples.tar')
