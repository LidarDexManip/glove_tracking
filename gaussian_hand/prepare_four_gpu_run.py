"""Create the explicitly requested four-GPU variant without changing audited data."""
import json

from gaussian_hand.diffusion_readiness import ROOT, PREPARED, FOUR_GPU_LAYOUT, digest, validate_ready


def main():
    config=json.loads(PREPARED.read_text())
    config['training'].update(batch_size=64,validation_batch_size=16,
        output_dir='outputs/hand_restoration/hugg_aria_gaussian_surface_v2_4gpu')
    config['comparison'].update(expected_world_size=4,execution_layout=FOUR_GPU_LAYOUT,
        prepared_config_sha256=digest(PREPARED),
        execution_note='User requested GPU 0-3. Same data, global batch, update budget and validation global batch; not an identical 8-rank numerical replay.')
    path=ROOT/'configs/hand_restoration/hugg_aria_gaussian_surface_v2_4gpu.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(config,indent=2)+'\n')
    ready=validate_ready(path,world_size=4)
    print(f'Four-GPU configuration verified: {ready["train_records"]} unchanged crops')


if __name__=='__main__':main()
