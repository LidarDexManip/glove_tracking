import numpy as np
import pytest

from temporal_demos.motion import context_windows,shared_ddpm_step
from temporal_demos.select_clips import fixed_box
from temporal_demos.render_clips import metrics
from temporal_demos.stabilize import stabilize_brightness


def test_stabilization_reduces_pulse_preserves_outside_and_real_exposure():
    base=np.full((9,32,32,3),100,np.uint8)
    pulse=base.copy();pulse[4]=120
    mask=np.zeros((9,32,32),bool);mask[:,4:28,4:28]=True
    fixed,report=stabilize_brightness(pulse,base/255,mask,mask)
    assert fixed[4,16,16,0]<110
    np.testing.assert_array_equal(fixed[~mask],pulse[~mask])
    unchanged,_=stabilize_brightness(pulse,pulse/255,mask,mask)
    np.testing.assert_array_equal(unchanged,pulse)
    assert max(abs(x) for x in report['correction_255'])<=24
    with pytest.raises(ValueError):stabilize_brightness(pulse,base/255,mask,mask,window=4)


def test_contexts_cover_every_frame_with_finite_overlap():
    for frames in (1,8,16,17,32,50,100):
        windows=context_windows(frames)
        assert set(i for w in windows for i in w)==set(range(frames))
        assert all(1<=len(w)<=16 for w in windows)
        assert all(w==list(range(w[0],w[-1]+1)) for w in windows)
        assert len({tuple(w) for w in windows})==len(windows)
    with pytest.raises(ValueError):context_windows(20,16,16)
    windows=context_windows(150,16,12)
    assert set(i for w in windows for i in w)==set(range(150))
    assert max(map(len,windows))==16
    for frames in (45,150):
        windows=context_windows(frames,24,16)
        assert set(i for w in windows for i in w)==set(range(frames))
        assert max(map(len,windows))==24
        assert all(len(set(a)&set(b))>=16 for a,b in zip(windows,windows[1:]))


def test_fixed_crop_contains_every_original_crop_and_stays_square():
    rows=[{'adaptive_crop_box_xyxy':[10,20,110,120]},
          {'adaptive_crop_box_xyxy':[15,18,125,128]}]
    x0,y0,x1,y1=fixed_box(rows)
    assert x1-x0==y1-y0
    for row in rows:
        a,b,c,d=row['adaptive_crop_box_xyxy']
        assert x0<=a and y0<=b and x1>=c and y1>=d


def test_temporal_metric_does_not_penalize_ground_truth_motion():
    rng=np.random.default_rng(7)
    frames=rng.integers(0,256,(3,32,32,3),dtype=np.uint8)
    masks=np.ones((3,32,32),bool)
    result=metrics(frames,frames,masks)
    assert result['visible_hand_mae_255']==0
    assert result['flow_compensated_temporal_residual_mae_255']==0
    assert result['brightness_flicker_mae_255']==0


def test_brightness_metric_detects_pulsing_without_penalizing_real_exposure():
    target=np.full((3,32,32,3),100,np.uint8)
    output=target.copy();output[1]=120
    masks=np.ones((3,32,32),bool)
    result=metrics(output,target,masks)
    assert abs(result['brightness_flicker_mae_255']-20)<1e-5
    matched=metrics(output,output,masks)
    assert matched['brightness_flicker_mae_255']==0


def test_shared_noise_ddpm_matches_single_frame_reference():
    import torch
    from diffusers import DDPMScheduler
    scheduler=DDPMScheduler(clip_sample=False,beta_schedule='scaled_linear',steps_offset=1)
    scheduler.set_timesteps(20)
    sample=torch.randn(1,4,1,8,8,generator=torch.Generator().manual_seed(1))
    prediction=torch.randn(1,4,1,8,8,generator=torch.Generator().manual_seed(2))
    for t in scheduler.timesteps:
        expected=scheduler.step(prediction,t,sample,generator=torch.Generator().manual_seed(7)).prev_sample
        actual=shared_ddpm_step(scheduler,prediction,t,sample,torch.Generator().manual_seed(7))
        torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-6)
        video=shared_ddpm_step(scheduler,prediction.repeat(1,1,3,1,1),t,
            sample.repeat(1,1,3,1,1),torch.Generator().manual_seed(7))
        torch.testing.assert_close(video,actual.repeat(1,1,3,1,1))
