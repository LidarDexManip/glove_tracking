import json
import os

import pytest

from gaussian_hand.archive_retention import (
    RETENTION_FILE, regression_archive, sha256, validate_archive_storage,
)
from gaussian_hand.compact_condition_archives import delete_full, selected_records, stat_signature


@pytest.fixture
def storage(tmp_path):
    d = tmp_path / 'image_sequences' / 'P0001_example'
    d.mkdir(parents=True)
    (d / 'images.tar').write_bytes(b'original')
    for name in ('samples.tar', 'samples.jsonl', 'index.jsonl', '_SUCCESS.json'):
        (d / name).write_bytes(b'retained')
    (tmp_path / 'READY.json').write_text(json.dumps({'rows': [
        {'sequence': d.name, 'archive_bytes': 8}]}))
    (tmp_path / 'dataset_audit.json').write_text('{}')
    entry = {'original_bytes': 8, 'original_stat': stat_signature(d / 'images.tar'),
             'record_indices': [0],
             'sha256': {n: sha256(d / n) for n in ('samples.tar','samples.jsonl','index.jsonl','_SUCCESS.json')}}
    marker = {'schema_version': 1, 'policy': 'regression_samples',
              'bindings': {n: sha256(tmp_path / n) for n in ('READY.json','dataset_audit.json')},
              'sequences': {d.name: entry}}
    (tmp_path / RETENTION_FILE).write_text(json.dumps(marker))
    return tmp_path, d


def test_missing_archive_requires_explicit_retention(storage):
    root, d = storage
    (root / RETENTION_FILE).unlink()
    assert validate_archive_storage(root) is None
    (d / 'images.tar').unlink()
    with pytest.raises(FileNotFoundError):
        validate_archive_storage(root)


def test_retained_mode_and_full_audit_are_distinct(storage):
    root, d = storage
    (d / 'images.tar').unlink()
    assert validate_archive_storage(root)['policy'] == 'regression_samples'
    assert regression_archive(root, d.name) == d / 'samples.tar'
    with pytest.raises(FileNotFoundError, match='full visual audit'):
        validate_archive_storage(root, require_full=True)


@pytest.mark.parametrize('name', ['samples.tar','samples.jsonl','index.jsonl','_SUCCESS.json'])
def test_changed_retained_artifact_rejected(storage, name):
    root, d = storage
    (d / name).write_bytes(b'changed!')
    with pytest.raises(ValueError, match='artifact changed'):
        validate_archive_storage(root)


def test_stale_binding_rejected(storage):
    root, d = storage
    (root / 'dataset_audit.json').write_text('{"changed": true}')
    with pytest.raises(ValueError, match='binding changed'):
        validate_archive_storage(root)


def test_delete_needs_live_audit_and_only_removes_original(storage):
    root, d = storage
    with pytest.raises(FileNotFoundError):
        delete_full(root)
    assert (d / 'images.tar').exists()
    (root / 'retained_sample_audit.json').write_text(json.dumps({
        'status': 'passed', 'retention_sha256': sha256(root / RETENTION_FILE)}))
    delete_full(root)
    assert not (d / 'images.tar').exists()
    assert (d / 'samples.tar').read_bytes() == b'retained'
    assert (d / 'index.jsonl').exists()
    assert json.loads((root / 'ARCHIVE_CLEANUP.json').read_text())['deleted_archives'] == 1


def test_changed_original_not_deleted(storage):
    root, d = storage
    (root / 'retained_sample_audit.json').write_text(json.dumps({
        'status': 'passed', 'retention_sha256': sha256(root / RETENTION_FILE)}))
    (d / 'images.tar').write_bytes(b'NEW_DATA')
    original = (d / 'images.tar').stat()
    os.utime(d / 'images.tar', ns=(original.st_atime_ns, original.st_mtime_ns + 1000000000))
    with pytest.raises(ValueError, match='Original changed'):
        delete_full(root)
    assert (d / 'images.tar').exists()


def test_selection_keeps_clip_frames_and_seen_examples():
    records = [{'sequence_id': 'P0001_f6cc0cc8', 'frame_index': i,
                'adaptive_handedness': 'right'} for i in range(1000, 1200)]
    chosen = selected_records(records, {('P0001_f6cc0cc8', 1007, 'right')})
    frames = {r['frame_index'] for r in chosen}
    assert set(range(1018,1168)) <= frames
    assert {1000,1007,1100,1199} <= frames
