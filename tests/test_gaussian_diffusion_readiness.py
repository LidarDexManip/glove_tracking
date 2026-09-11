import copy
import json
from pathlib import Path

import pytest

from gaussian_hand.diffusion_readiness import validate_comparison


def config():
    return json.loads(Path('configs/hand_restoration/hugg_aria_gaussian_surface_strict_v2.json').read_text())


def test_strict_configuration_preserves_baseline():
    validate_comparison(config())


@pytest.mark.parametrize('section,key,value',[
    ('training','batch_size',64),('training','gradient_accumulation_steps',2),
    ('training','max_train_steps',20000),('data','crop_scale',1.35),
    ('data','loss_mask_source','sam'),('model','conditioning_mode','rgb'),
    ('inference','steps',50)])
def test_comparison_rejects_uncontrolled_changes(section,key,value):
    altered=config();altered[section][key]=value
    with pytest.raises(ValueError):validate_comparison(altered)


def test_training_entrypoint_checks_readiness_before_accelerator(monkeypatch):
    import sys
    import accelerate
    import gaussian_hand.diffusion_readiness as gate
    import train_hand_restorer
    monkeypatch.setenv('HF_HUB_OFFLINE','1')
    monkeypatch.setenv('TRANSFORMERS_OFFLINE','1')
    monkeypatch.setattr(sys,'argv',['train_hand_restorer.py','--config',
        'configs/hand_restoration/hugg_aria_gaussian_surface_strict_v2.json'])
    def refuse(*args,**kwargs):raise RuntimeError('test readiness gate')
    def forbidden(*args,**kwargs):raise AssertionError('GPU/model setup must not run before the gate')
    monkeypatch.setattr(gate,'validate_ready',refuse)
    monkeypatch.setattr(accelerate,'Accelerator',forbidden)
    with pytest.raises(RuntimeError,match='test readiness gate'):train_hand_restorer.main()


def test_launcher_is_read_only_without_execute(monkeypatch,capsys):
    import sys
    import gaussian_hand.launch_diffusion_strict as launcher
    monkeypatch.setattr(sys,'argv',['launcher'])
    monkeypatch.setattr(launcher,'validate_ready',lambda path:{'train_records':524917})
    def forbidden(*args,**kwargs):raise AssertionError('Training must not launch by default')
    monkeypatch.setattr(launcher.subprocess,'run',forbidden)
    launcher.main()
    assert 'No training started' in capsys.readouterr().out
