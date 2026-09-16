import numpy as np
import pytest

from temporal_demos.motion import context_windows
from temporal_demos.select_clips import fixed_box
from temporal_demos.render_clips import metrics


def test_contexts_cover_every_frame_with_finite_overlap():
    for frames in (1,8,16,17,32,50,100):
        windows=context_windows(frames)
        assert set(i for w in windows for i in w)==set(range(frames))
        assert all(1<=len(w)<=16 for w in windows)
        assert all(w==list(range(w[0],w[-1]+1)) for w in windows)
        assert len({tuple(w) for w in windows})==len(windows)
    with pytest.raises(ValueError):context_windows(20,16,16)


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
