"""Run the existing bare-hand trainer with the prepared clinical crop adapter.

Shared training code and historical configurations are left intact.
"""
import json
from pathlib import Path
import torch
import train_hand_restorer as trainer
from hand_restoration.diffusion import ControlNetHandRestorer
from gaussian_glove.task_overlay_dataset import TaskOverlayDataset

class TaskRestorer(ControlNetHandRestorer):
    @torch.no_grad()
    def text_embeddings(self,batch_size,prompt=None):
        # Empty negative text must remain empty, not fall back to positive text.
        text=self.config.prompt if prompt is None else prompt
        tokens=self.tokenizer([text]*batch_size,padding='max_length',
            max_length=self.tokenizer.model_max_length,truncation=True,return_tensors='pt').input_ids.to(self.device_name)
        return self.text_encoder(tokens)[0]

def make_prepared(config,manifest):
    if config['data'].get('cache_kind')!='task_overlay_png_v1': raise ValueError('Wrong dataset adapter')
    ready=json.loads((Path(manifest).parent/'READY.json').read_text())
    if ready['source_frames']!=6300 or len(ready['sessions'])!=14: raise ValueError('Incomplete preparation')
    return TaskOverlayDataset(manifest)

def main():
    trainer.make_derived_dataset=make_prepared
    trainer.ControlNetHandRestorer=TaskRestorer
    trainer.main()

if __name__=='__main__': main()
