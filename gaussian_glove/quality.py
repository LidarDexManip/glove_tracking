"""Deterministic image/geometry gates. Missing flags never imply manual QA."""
import csv
import cv2
import numpy as np
DEFAULTS = {'min_visible_vertices': .98, 'min_iou': .55, 'min_mesh_coverage': .80,
            'min_mask_supported': .65, 'min_largest_component': .90,
            'max_hole_fraction': .05, 'max_area_ratio': 2.0,
            'normal_offset_max_m': .008}

def load_exceptions(path):
    with path.open(newline='', encoding='utf-8-sig') as f: return list(csv.DictReader(f))

def exception_reasons(rows, session, side, frame):
    reasons=[]
    for r in rows:
        component=r['component']
        relevant=component in ('rgb','rgb_timing','mano','mask',f'mano_{side}',f'mask_{side}')
        if r['session']==session and relevant and int(r['frame_start'])<=frame<=int(r['frame_end']) and r['status'] in ('confirmed_invalid','review_required'):
            reasons.append('quality_index:'+r['issue'])
    return reasons

def silhouette(uv, depth, faces, width, height):
    result=np.zeros((height,width),np.uint8)
    valid=np.isfinite(uv).all(1)&np.isfinite(depth)&(depth>.01)
    for polygon in np.rint(uv[faces[valid[faces].all(1)]]).astype(np.int32):
        cv2.fillConvexPoly(result,polygon,1)
    return result.astype(bool)

def evaluate(mask, mesh, radius, policy, neighbor_areas=()):
    mask=mask.astype(bool);mesh=mesh.astype(bool);reasons=[];area=int(mask.sum());mesh_area=int(mesh.sum())
    if not area:return ['empty_mask'],{},np.ones(mask.shape,bool)
    if not mesh_area:return ['empty_mesh'],{},np.ones(mask.shape,bool)
    intersection=int((mask&mesh).sum());iou=intersection/max(int((mask|mesh).sum()),1)
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1))
    support=cv2.dilate(mesh.astype(np.uint8),kernel).astype(bool)
    count,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
    largest=int(stats[1:,cv2.CC_STAT_AREA].max())/area
    contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    filled=np.zeros(mask.shape,np.uint8);cv2.drawContours(filled,contours,-1,1,-1)
    holes=float((filled.astype(bool)&~mask).sum()/max(int(filled.sum()),1))
    metrics={'iou':iou,'mesh_coverage':intersection/mesh_area,'mask_supported':int((mask&support).sum())/area,'largest_component':largest,'hole_fraction':holes,'mask_area':area}
    for key in ('iou','mesh_coverage','mask_supported','largest_component'):
        if metrics[key]<policy['min_'+key]:reasons.append('low_'+key)
    if holes>policy['max_hole_fraction']:reasons.append('mask_holes')
    if len(neighbor_areas)==2 and min(neighbor_areas)>0:
        a,b=neighbor_areas
        if max(a,b)/min(a,b)<1.3 and (area<min(a,b)/policy['max_area_ratio'] or area>max(a,b)*policy['max_area_ratio']):reasons.append('isolated_area_jump')
    # Ignore glove pixels beyond the bounded surface neighborhood (mostly cuff).
    # All background stays supervised; offsets still receive real silhouette loss.
    valid=~(mask&~support)
    return reasons,metrics,valid
