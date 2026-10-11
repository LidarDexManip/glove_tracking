"""Quality and offset regression tests; CPU only unless explicitly run elsewhere."""
import csv,json,hashlib
from pathlib import Path
import cv2,numpy as np,pytest,torch
from gaussian_glove.quality import DEFAULTS,evaluate,exception_reasons
from gaussian_hand.losses import reconstruction_loss
from gaussian_hand.data.dataset import GaussianHandDataset

def test_quality_rejects_empty_misaligned_and_fragmented_masks():
    mesh=np.zeros((64,64),bool);mesh[16:48,16:48]=True
    assert 'empty_mask' in evaluate(np.zeros_like(mesh),mesh,3,DEFAULTS)[0]
    displaced=np.roll(mesh,27,axis=1)
    assert 'low_iou' in evaluate(displaced,mesh,3,DEFAULTS)[0]
    assert evaluate(mesh,mesh,3,DEFAULTS)[0]==[]
    fragmented=mesh.copy();fragmented[:,31:37]=False
    assert 'low_largest_component' in evaluate(fragmented,mesh,3,DEFAULTS)[0]

def test_quality_index_side_and_frame_boundaries():
    rows=[{'session':'session_041','frame_start':'365','frame_end':'365','component':'rgb','status':'confirmed_invalid','issue':'splice'}, {'session':'s','frame_start':'2','frame_end':'4','component':'mano_left','status':'review_required','issue':'drift'}]
    assert exception_reasons(rows,'session_041','right',365)
    assert not exception_reasons(rows,'session_041','right',364)
    assert exception_reasons(rows,'s','left',4)
    assert not exception_reasons(rows,'s','right',4)

def test_cuff_ignored_but_background_and_thickness_supervised():
    mesh=np.zeros((64,64),bool);mesh[16:48,16:48]=True
    mask=mesh.copy();mask[48:63,24:40]=True
    _,_,valid=evaluate(mask,mesh,3,DEFAULTS)
    assert not valid[61,30] and valid[50,30] and valid[0,0]
    image=torch.ones(64,64,3);target=torch.tensor(mask[...,None],dtype=torch.float32)
    rgb=torch.zeros_like(image,requires_grad=True);alpha=torch.zeros(64,64,1,requires_grad=True)
    loss,_=reconstruction_loss({'rgb':rgb,'alpha':alpha,'distortion':torch.zeros_like(alpha)},image,target,supervision=torch.tensor(valid[...,None],dtype=torch.float32))
    loss.backward()
    assert alpha.grad[50,30].abs().item()>0
    assert alpha.grad[61,30].item()==0

def test_unqualified_record_cannot_enter_training(tmp_path):
    root=tmp_path/'left';root.mkdir();row={'split':'train','quality_pass':False,'mask_source':'hf_sam3.1','source_revision':'rev','sequence':'black_glove_s','supervision':'mask.png'}
    manifest=root/'frames.jsonl';manifest.write_text(json.dumps(row)+'\n');(root/'summary.json').write_text(json.dumps({'frames_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'source_revision':'rev'}));(tmp_path/'preparation.json').write_text('{"complete":true}')
    with pytest.raises(ValueError,match='Unqualified'):GaussianHandDataset(root,require_quality=True)

def test_offset_is_bounded_differentiable_and_legacy_still_zero():
    from gaussian_hand.model import GaussianHand
    root=Path(__file__).resolve().parents[1]
    model=GaussianHand(str(root/'mano_v1_2/models/MANO_RIGHT.pkl'),torch.zeros(10),gaussians_per_face=1,normal_offset_max=.008,strict_surface=False,pose_representation='axis_angle',bounded_scales=True)
    pose=torch.zeros(45);q=torch.tensor([1.,0,0,0]);t=torch.tensor([0.,0,.5]);a=model.attributes(pose,q,t)
    a['means'].square().sum().backward()
    assert torch.isfinite(model.normal_offset_logits.grad).all() and model.normal_offset_logits.grad.abs().sum()>0
    offset=torch.sigmoid(model.normal_offset_logits)*model.normal_offset_max
    assert (offset>=0).all() and (offset<=.008).all()
    legacy=GaussianHand(str(root/'mano_v1_2/models/MANO_RIGHT.pkl'),torch.zeros(10),gaussians_per_face=1,normal_offset_max=.008,strict_surface=True)
    assert legacy.normal_offset_max==0
