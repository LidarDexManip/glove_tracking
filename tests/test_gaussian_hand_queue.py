import json

from gaussian_hand.run_training_queue import preview_window


def test_preview_uses_longest_contiguous_run(tmp_path):
    manifest = tmp_path / "frames.jsonl"
    records = [{"sequence": "P0001_b", "frame_index": i} for i in range(100, 180)]
    records += [{"sequence": "P0001_a", "frame_index": i} for i in (1, 3, 5)]
    manifest.write_text("\n".join(json.dumps(r) for r in reversed(records)))
    assert preview_window(manifest) == ("P0001_b", 110, 60)


def test_preview_does_not_bridge_sequences(tmp_path):
    manifest = tmp_path / "frames.jsonl"
    records = [{"sequence": "P0001_a", "frame_index": 9},
               {"sequence": "P0001_b", "frame_index": 10},
               {"sequence": "P0001_b", "frame_index": 11}]
    manifest.write_text("\n".join(json.dumps(r) for r in records))
    assert preview_window(manifest) == ("P0001_b", 10, 2)
