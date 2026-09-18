import numpy as np
import pytest

from hand_restoration.hugg_aria_conditioning import CropBox
from temporal_demos.render_two_hands import composite_pair


def test_both_hands_are_edited_and_background_unchanged():
    base=np.full((32,32,3),.2,np.float32)
    mask=np.ones((8,8),bool)
    condition=[np.full((8,8,3),v,np.float32) for v in (.4,.6)]
    restored=[np.full((8,8,3),v,np.float32) for v in (.5,.7)]
    boxes=[CropBox(2,2,10,10),CropBox(20,20,28,28)]
    a,b,masks=composite_pair(base,condition,restored,[mask,mask],boxes)
    np.testing.assert_allclose(a[3,3],.4);np.testing.assert_allclose(a[22,22],.6)
    np.testing.assert_allclose(b[3,3],.5);np.testing.assert_allclose(b[22,22],.7)
    np.testing.assert_array_equal(b[~(masks[0]|masks[1])],base[~(masks[0]|masks[1])])
    with pytest.raises(ValueError):composite_pair(base,condition,restored,[mask,mask],[boxes[0],boxes[0]])
    with pytest.raises(ValueError):composite_pair(base,condition,restored,[mask,np.zeros_like(mask)],boxes)
