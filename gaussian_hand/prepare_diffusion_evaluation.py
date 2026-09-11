"""Freeze paired visual evaluation selections now; no model inference/training."""
import json

import torch

from gaussian_hand.prepare_diffusion_strict import CONFIG, OUT, ROOT, digest


def main():
    freeze = json.loads((ROOT / 'baselines/adaptive_splitcrop_v1/BASELINE_FREEZE.json').read_text())
    config = json.loads(CONFIG.read_text())
    demo = ROOT / freeze['demo']['contact_sheet']
    manifest = demo.parent / 'manifest.json'
    if digest(manifest) != freeze['demo']['inference_manifest_sha256']:
        raise ValueError('Historical visual evaluation manifest changed')
    prior = json.loads(manifest.read_text())
    train = [json.loads(line) for line in (ROOT / config['data']['train_manifest']).open()]
    seen = [json.loads(line) for line in (ROOT / config['data']['val_manifest']).open()]
    for sample in prior['samples']:
        record = train[sample['adaptive_train_manifest_index']]
        assert record['sequence_id'] == sample['sequence_id']
        assert record['frame_index'] == sample['frame_id']
        assert record['adaptive_handedness'] == sample['handedness']
    subset = torch.randperm(len(seen), generator=torch.Generator().manual_seed(7))[:136].sort().values.tolist()
    plan = {
        'status': 'prepared_not_executed', 'training_started': False,
        'config_sha256': digest(CONFIG),
        'historical_demo_manifest_sha256': digest(manifest),
        'historical_same10': prior['samples'],
        'primary_visual_evaluation': 'all 183 frozen seen-eval crops; retain hard cases',
        'seen_eval_records': seen,
        'training_validation_subset_indices': subset,
        'inference': config['inference'],
        'same10_seed_per_sample': 7,
        'seen_seed_rule': '7 + seen_eval_record_index; identical seed for the two arms',
        'checkpoint_policy': 'compare final 15,380-step checkpoints; no best-checkpoint selection',
        'arms': {
            'baseline': {'config': freeze['configuration']['path'],
                         'checkpoint': freeze['training_result']['checkpoint'],
                         'checkpoint_sha256': freeze['training_result']['checkpoint_sha256']},
            'gaussian': {'config': str(CONFIG.relative_to(ROOT)),
                         'checkpoint': config['training']['output_dir'] + '/controlnet_final.pt'}},
        'display_policy': 'same source frame and crop; compare raw model outputs plus identical-mask paste-backs separately',
        'metric_policy': 'report on the same frozen MANO mask and target; do not compare differently masked scores',
        'scope': 'seen-data reconstruction, not held-out generalization',
        'dataset_factory': 'train_hand_restorer.make_hugg_aria_dataset(include_numpy=True)',
    }
    (OUT / 'evaluation_plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    print('Frozen historical same10 and all 183 seen crops for paired evaluation. No inference or training started.')


if __name__ == '__main__':
    main()
