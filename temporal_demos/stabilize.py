"""Offline photometric stabilization, with no frame blending or target RGB access."""
import cv2
import numpy as np


def stabilize_brightness(frames, conditions, visible_masks, edit_masks,
                         window=9, strength=.9, max_correction=24.):
    if window < 1 or window % 2 != 1 or not 0 <= strength <= 1:
        raise ValueError('Use an odd positive window and strength in [0,1]')
    if not (len(frames) == len(conditions) == len(visible_masks) == len(edit_masks)):
        raise ValueError('Frame and mask counts must match')
    luma=np.array([.2126,.7152,.0722],np.float32)
    biases=[]
    for frame,condition,visible in zip(frames,conditions,visible_masks):
        mask=cv2.erode(np.asarray(visible,np.uint8),np.ones((5,5),np.uint8)).astype(bool)
        if not mask.any():mask=np.asarray(visible,bool)
        residual=(frame.astype(np.float32)-np.asarray(condition,np.float32)*255)@luma
        biases.append(float(residual[mask].mean()) if mask.any() else np.nan)
    bias=np.asarray(biases)
    valid=np.isfinite(bias)
    if not valid.any():return np.array(frames,copy=True),dict(correction_255=[0.]*len(frames))
    bias=np.interp(np.arange(len(bias)),np.flatnonzero(valid),bias[valid])
    radius=window//2
    weights=np.exp(-.5*(np.arange(-radius,radius+1)/max(window/4,1))**2)
    weights/=weights.sum()
    smooth=np.convolve(np.pad(bias,(radius,radius),mode='edge'),weights,mode='valid')
    correction=np.clip((smooth-bias)*strength,-max_correction,max_correction)
    output=[]
    for frame,edit,offset in zip(frames,edit_masks,correction):
        # Feather strictly inside the edit mask. Never introduce outside-mask changes.
        feather=np.minimum(cv2.distanceTransform((np.asarray(edit)>0).astype(np.uint8),
                                                cv2.DIST_L2,3)/5.,1.)
        output.append(np.clip(np.rint(frame.astype(np.float32)+offset*feather[...,None]),0,255).astype(np.uint8))
    return np.asarray(output),dict(window=window,strength=strength,max_correction_255=max_correction,
        correction_255=correction.tolist(),input_bias_255=bias.tolist(),
        method='Gaussian-smoothed output-minus-condition mean luminance; no RGB target, no frame blending')
