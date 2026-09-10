import numpy as np
import pytest

from gaussian_hand.make_subject_result_video import restore_render


def test_restore_places_rgb_at_source_coordinates_without_double_alpha():
    rgb=np.full((2,2,3),100,np.uint8)
    alpha=np.full((2,2),128,np.uint8)
    image=restore_render(rgb,alpha,(8,8,3),(2,3,4,5))
    assert (image[3:5,2:4]==100).all()
    assert image.sum()==2*2*3*100


def test_restore_removes_rgb_codec_noise_outside_alpha():
    image=restore_render(np.full((2,2,3),5,np.uint8),np.zeros((2,2),np.uint8),
                         (8,8,3),(1,1,3,3))
    assert not image.any()


def test_restore_rejects_out_of_bounds_crop():
    with pytest.raises(ValueError):
        restore_render(np.zeros((2,2,3),np.uint8),np.zeros((2,2),np.uint8),
                       (8,8,3),(0,0,9,8))
