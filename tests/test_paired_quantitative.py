import numpy as np

from scripts.evaluation.quantify_paired_restorers import region_metrics


def test_identical_hand_has_zero_error_and_unit_ssim():
    target = np.random.default_rng(4).random((32, 32, 3)).astype(np.float32)
    mask = np.zeros((32, 32), bool)
    mask[4:28, 4:28] = True
    result = region_metrics(target, target, mask)
    assert result['mse'] == 0
    assert result['mae_255'] == 0
    assert result['ssim_interior'] == 1
    assert result['pixels'] == 24*24
    assert result['ssim_pixels'] == 14*14


def test_background_changes_do_not_improve_or_damage_hand_metrics():
    target = np.full((32, 32, 3), .5, np.float32)
    mask = np.zeros((32, 32), bool)
    mask[4:28, 4:28] = True
    prediction = np.zeros_like(target)
    prediction[mask] = target[mask]
    result = region_metrics(prediction, target, mask)
    assert result['mse'] == 0
    assert result['mae_255'] == 0
    assert abs(result['ssim_interior']-1) < 1e-10


def test_known_pixel_error_and_empty_region():
    target = np.zeros((32, 32, 3), np.float32)
    prediction = np.full_like(target, .1)
    mask = np.ones((32, 32), bool)
    result = region_metrics(prediction, target, mask)
    assert abs(result['psnr_db']-20) < 1e-5
    assert abs(result['mae_255']-25.5) < 1e-5
    assert region_metrics(prediction,target,~mask) is None
    tiny = np.zeros_like(mask)
    tiny[10,10] = True
    assert region_metrics(prediction,target,tiny)['ssim_interior'] is None
