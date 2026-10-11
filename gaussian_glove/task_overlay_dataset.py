"""Read the prepared lossless overlay crop cache without rebuilding conditions."""
import json, tarfile
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class TaskOverlayDataset(Dataset):
    def __init__(self, manifest):
        self.manifest_path=Path(manifest); self.root=self.manifest_path.parent
        self.samples=[json.loads(x) for x in self.manifest_path.read_text().splitlines() if x.strip()]
        if not self.samples: raise ValueError('Empty prepared dataset')
        self._handles={}

    def __len__(self): return len(self.samples)

    def __getstate__(self):
        state=self.__dict__.copy(); state['_handles']={}; return state

    def close(self):
        for handle in self._handles.values(): handle.close()
        self._handles={}

    def __del__(self):
        if hasattr(self,'_handles'): self.close()

    def read_array(self, row, member, rgb=False):
        path=self.root/row['shard']
        if path not in self._handles: self._handles[path]=tarfile.open(path,'r')
        stream=self._handles[path].extractfile(row['key']+'.'+member+'.png')
        if stream is None: raise KeyError(member)
        with stream: data=np.frombuffer(stream.read(),np.uint8)
        image=cv2.imdecode(data,cv2.IMREAD_COLOR if rgb else cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape[:2]!=(512,512): raise ValueError('Bad cached image')
        if rgb: image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        return image.astype(np.float32)/255

    def __getitem__(self, index):
        row=self.samples[index]
        def rgb(name): return torch.from_numpy((self.read_array(row,name,True)*2-1).transpose(2,0,1).copy())
        def mask(name): return torch.from_numpy(self.read_array(row,name)[None].copy())
        return {'target_rgb':rgb('target'),'condition_rgb':rgb('condition'),
                'condition_mask':mask('visible_mask'),'loss_mask':mask('mano_mask'),
                'edit_mask':mask('edit_mask'),'metadata':{'sequence_id':row['sequence_id'],
                    'frame_id':row['frame_id'],'handedness':row['handedness'],
                    'camera_id':row['camera_id']}}
