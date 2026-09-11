"""CPU tests for matched-frame planning and safe checkpoint selection."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from hand_restoration.paired_video import (
    FrameCatalog, cache_key, checkpoint_choices, read_configs, resolve_checkpoint,
)


def record(frame, hand='right', area=1):
    return dict(sequence_id='test', frame_index=frame,
                adaptive_handedness=hand, adaptive_component_area=area)


class PairedVideoTests(unittest.TestCase):
    def test_auto_is_deterministic_and_preserves_original_rows(self):
        rows = [record(0, 'left', 20), record(0, 'right', 40),
                record(0, 'both', 1), record(6, 'left', 20), record(6, 'right', 40)]
        catalog = FrameCatalog(rows)
        self.assertEqual(catalog.selected('test', 'auto'), {0: 2, 6: 4})
        result = catalog.plan('test', 0, 1, 1, 'right')
        self.assertEqual(result, [dict(rows[1], global_record_index=1)])
        self.assertNotIn('global_record_index', rows[1])

    def test_timeline_and_missing_crop_fail_closed(self):
        catalog = FrameCatalog([record(f) for f in range(100)])
        self.assertEqual([r['frame_index'] for r in catalog.plan('test', 10, 1, 5, 'right')],
                         [10, 16, 22, 28, 34])
        with self.assertRaises(ValueError):
            catalog.plan('test', 10, 1, 5, 'left')
        with self.assertRaises(ValueError):
            catalog.plan('test', 90, 1, 5, 'right')
        gaps = FrameCatalog([record(f) for f in (0, 6, 18, 24)])
        with self.assertRaises(ValueError):
            gaps.plan('test', 0, 1, 5, 'auto')
        self.assertEqual(catalog.next_start('test', 95, 1, 5, 'right'), 0)

    def test_invalid_sampling(self):
        for duration, fps in [(0, 5), (11, 5), (1, 7), (float('nan'), 5), (1.1, 5)]:
            with self.assertRaises(ValueError):
                FrameCatalog.offsets(duration, fps)
        catalog = FrameCatalog([record(0)])
        for start in (-1, .5, float('nan')):
            with self.assertRaises(ValueError):
                catalog.plan('test', start, 1, 1, 'right')

    def test_checkpoints_order_integrity_and_arm_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = ['controlnet_final.pt', 'controlnet_step000100.pt', 'controlnet_step000020.pt']
            for name in names:
                with zipfile.ZipFile(root/name, 'w') as archive:
                    archive.writestr('checkpoint/data.pkl', b'test')
            (root/'controlnet_step999999.pt').write_bytes(b'partial archive')
            other = root/'other'
            other.mkdir()
            with zipfile.ZipFile(other/'controlnet_step999998.pt', 'w') as archive:
                archive.writestr('checkpoint/data.pkl', b'test')
            (root/'controlnet_step999998.pt').symlink_to(other/'controlnet_step999998.pt')
            choices = checkpoint_choices(root, minimum_age=0, minimum_size=0)
            self.assertEqual([p.name for p in choices], names)
            with patch('hand_restoration.paired_video.checkpoint_choices', return_value=choices):
                self.assertEqual(resolve_checkpoint(root/names[0], root), root/names[0])
                with self.assertRaises(ValueError):
                    resolve_checkpoint(other/'controlnet_step999998.pt', root)

    def test_configs_preserve_data_comparability(self):
        baseline, gaussian = read_configs()
        self.assertNotEqual(baseline['data']['render_root'], gaussian['data']['render_root'])
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory)/name for name in ('baseline.json', 'gaussian.json')]
            gaussian['data']['train_manifest'] = 'different.jsonl'
            for path, config in zip(paths, (baseline, gaussian)):
                path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                read_configs(*paths)

    def test_cache_identity(self):
        recipe = dict(checkpoints=['a', 'b'], records=[record(0)], seed=7)
        self.assertEqual(cache_key(recipe), cache_key(dict(reversed(list(recipe.items())))))
        self.assertNotEqual(cache_key(recipe), cache_key(dict(recipe, seed=8)))
        self.assertNotEqual(cache_key(recipe), cache_key(dict(recipe, checkpoints=['a', 'c'])))

    def test_ui_build_does_not_load_gpu(self):
        from compare_hugg_restorers_web import build_ui
        class FakeApp:
            catalog = FrameCatalog([record(f) for f in range(100)])
            gpu, allow_shared_gpu = 0, False
            def choices(self, arm):
                return [('controlnet_final.pt', f'/arm{arm}/controlnet_final.pt')]
            def summary(self, *args):
                return 'Ready'
        with patch('subprocess.Popen', side_effect=AssertionError('Must not launch GPU worker')):
            ui = build_ui(FakeApp())
        self.assertIn('compare_video', [d.get('api_name') for d in ui.config['dependencies']])


if __name__ == '__main__':
    unittest.main()
