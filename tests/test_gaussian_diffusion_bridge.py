import numpy as np
import pytest
import torch

from hand_restoration.hugg_aria_conditioning import build_faithful_condition
from gaussian_hand.render_scene import scaled_intrinsics, render_scene


def test_straight_and_premultiplied_conditions_are_equivalent():
    rng=np.random.default_rng(5)
    target=rng.random((32,32,3),dtype=np.float32)
    color=rng.random(target.shape,dtype=np.float32)
    alpha=rng.random((32,32),dtype=np.float32)
    alpha[:4]=0
    mano=np.ones(alpha.shape,bool);visible=mano.copy();visible[:,20:]=False
    kwargs=dict(visible_hand_mask=visible,wrist_enabled=False,dilation_px=0,render_opacity=.7)
    straight=build_faithful_condition(target,color,alpha,mano,**kwargs)
    premult=build_faithful_condition(target,color*alpha[...,None],alpha,mano,
                                    render_rgb_semantics="premultiplied",**kwargs)
    for key in ("raw_overlay_rgb","overlay_rgb","condition_rgb","edit_mask","visible_hand_mask"):
        np.testing.assert_allclose(getattr(straight,key),getattr(premult,key),atol=1e-7)


def test_no_target_hand_leaks_through_transparent_gaussian_holes():
    target=np.ones((32,32,3),np.float32)
    alpha=np.zeros((32,32),np.float32);alpha[10:20,10:20]=.5
    color=np.full_like(target,.8)*alpha[...,None]
    mano=np.zeros(alpha.shape,bool);mano[5:25,5:25]=True
    kw=dict(wrist_enabled=False,dilation_px=0,render_rgb_semantics="premultiplied")
    a=build_faithful_condition(target,color,alpha,mano,**kw)
    changed=target.copy();changed[mano]=.1
    b=build_faithful_condition(changed,color,alpha,mano,**kw)
    np.testing.assert_array_equal(a.condition_rgb,b.condition_rgb)
    assert (a.condition_rgb[6,6]==0).all()
    np.testing.assert_allclose(a.condition_rgb[12,12],.4)


def test_masked_out_instance_rgb_does_not_leak():
    target=np.zeros((8,8,3),np.float32)
    result=build_faithful_condition(target,np.ones_like(target),np.zeros((8,8)),
        np.ones((8,8),bool),wrist_enabled=False,render_rgb_semantics="premultiplied")
    assert not result.condition_rgb.any()


def test_full_frame_intrinsics_match_pixel_center_resize():
    k=scaled_intrinsics({"image_width":1024,"image_height":1024,
                         "focal_lengths":[800,800],"principal_point":[511.5,511.5]})
    np.testing.assert_allclose(k,[[400,0,255.5],[0,400,255.5],[0,0,1]])


def test_empty_scene_is_transparent_on_cpu():
    r=render_scene([],torch.eye(4),torch.eye(3),size=8)
    assert not r["rgb"].any() and not r["alpha"].any()
