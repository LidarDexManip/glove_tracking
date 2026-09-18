"""Inference-only AnimateDiff for our trained five-channel latent-mask ControlNet."""
from __future__ import annotations

from dataclasses import asdict

import numpy as np
import torch

from hand_restoration.inference import build_restorer

ADAPTER='guoyww/animatediff-motion-adapter-v1-5-3'


def context_windows(frames, size=16, overlap=8):
    if frames<1 or size<1 or not 0<=overlap<size:
        raise ValueError('Invalid temporal context geometry')
    if frames<=size:return [list(range(frames))]
    starts=list(range(0,frames-size+1,size-overlap))
    if starts[-1]!=frames-size:starts.append(frames-size)
    return [list(range(start,start+size)) for start in starts]


def shared_ddpm_step(scheduler,prediction,timestep,sample,generator):
    """Diffusers' epsilon/fixed-small posterior, with one noise field per clip.

    The marginal per-frame posterior is unchanged. Only cross-frame noise
    covariance changes. Single-frame output matches DDPMScheduler.step.
    """
    if (scheduler.config.prediction_type!='epsilon' or scheduler.config.variance_type!='fixed_small'
            or scheduler.config.clip_sample or scheduler.config.thresholding or sample.ndim!=5):
        raise ValueError('Shared DDPM supports unclipped epsilon/fixed_small BCFHW only')
    t=timestep;previous=scheduler.previous_timestep(t)
    alpha=scheduler.alphas_cumprod[t]
    alpha_previous=scheduler.alphas_cumprod[previous] if previous>=0 else scheduler.one
    beta=1-alpha;beta_previous=1-alpha_previous
    current_alpha=alpha/alpha_previous;current_beta=1-current_alpha
    predicted_clean=(sample-beta.sqrt()*prediction)/alpha.sqrt()
    result=(alpha_previous.sqrt()*current_beta/beta)*predicted_clean
    result=result+(current_alpha.sqrt()*beta_previous/beta)*sample
    if t>0:
        noise=torch.randn((sample.shape[0],sample.shape[1],1,*sample.shape[-2:]),
                          generator=generator,device=sample.device,dtype=sample.dtype)
        result=result+scheduler._get_variance(t).sqrt()*noise
    return result


class TemporalRestorer:
    def __init__(self,config,checkpoint,device='cuda',dtype=torch.bfloat16):
        from diffusers import MotionAdapter,UNetMotionModel
        self.config=config
        self.model=build_restorer(config,device=device)
        payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if payload['config']!=asdict(self.model.config):raise ValueError('Checkpoint/config mismatch')
        self.model.controlnet.load_state_dict(payload['state_dict'],strict=True)
        self.checkpoint_step=int(payload['global_step'])
        del payload
        self.model.eval().requires_grad_(False).to(dtype=dtype)
        adapter=MotionAdapter.from_pretrained(ADAPTER,local_files_only=True,torch_dtype=dtype)
        self.motion=UNetMotionModel.from_unet2d(self.model.unet,motion_adapter=adapter).to(device=device,dtype=dtype)
        self.motion.eval().requires_grad_(False)
        # Spatial UNet weights are copied unchanged; the adapter is a separate module.
        del adapter
        self.model.unet.to('cpu')
        self.dtype,self.device=dtype,device
        self.strength=1.
        self.motion_layers=0
        for name,module in self.motion.named_modules():
            if name.rsplit('.',1)[0].endswith('.motion_modules') and name.rsplit('.',1)[-1].isdigit():
                original=module.forward
                def scaled_forward(hidden_states,*args,_original=original,**kwargs):
                    if self.strength==0:return hidden_states
                    value=_original(hidden_states,*args,**kwargs)
                    return value if self.strength==1 else hidden_states+self.strength*(value-hidden_states)
                module.forward=scaled_forward
                self.motion_layers+=1
        if not self.motion_layers:raise RuntimeError('No motion layers found')
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def encode(self,conditions,masks):
        outputs=[]
        for start in range(0,len(conditions),4):
            c=torch.from_numpy(np.stack(conditions[start:start+4])).permute(0,3,1,2)
            c=(c*2-1).to(device=self.device,dtype=self.dtype)
            m=torch.from_numpy(np.stack(masks[start:start+4])).unsqueeze(1).to(device=self.device,dtype=self.dtype)
            outputs.append(self.model._control_condition(c,m))
        return torch.cat(outputs)

    @torch.inference_mode()
    def generate(self,encoded,*,strength=1.,steps=20,seed=7,progress=None,
                 scheduler_kind='original',initial_rgb=None,denoise_strength=.35,
                 context_size=16,overlap=8):
        if not 0<=strength<=1:raise ValueError('Motion strength must be 0..1')
        self.strength=float(strength)
        frames=encoded.shape[0]
        if context_size>32:raise ValueError('Adapter positional encoding supports at most 32 frames')
        windows=context_windows(frames,context_size,overlap)
        if scheduler_kind in ('original','ddpm-shared'):
            scheduler=self.model.noise_scheduler
        elif scheduler_kind in ('ddim','ddim-linear','ddim-stable'):
            from diffusers import DDIMScheduler
            kwargs=dict(clip_sample=False,timestep_spacing='linspace')
            if scheduler_kind=='ddim-linear':kwargs['beta_schedule']='linear'
            if scheduler_kind=='ddim-stable':kwargs['timestep_spacing']='leading'
            scheduler=DDIMScheduler.from_config(self.model.noise_scheduler.config,**kwargs)
        else:raise ValueError('Unknown scheduler')
        scheduler.set_timesteps(steps,device=self.device)
        generator=torch.Generator(device=self.device).manual_seed(seed)
        # Same initial noise across frames. Both on/off modes use identical random draws.
        noise=torch.randn((1,4,1,*encoded.shape[-2:]),generator=generator,
                          device=self.device,dtype=self.dtype).repeat(1,1,frames,1,1)
        timesteps=scheduler.timesteps
        if initial_rgb is None:
            latents=noise*scheduler.init_noise_sigma
        else:
            if not 0<denoise_strength<=1:raise ValueError('Invalid video-to-video strength')
            initial=[]
            for start in range(0,frames,4):
                rgb=torch.from_numpy(initial_rgb[start:start+4]).permute(0,3,1,2).to(self.device,self.dtype)/127.5-1
                initial.append(self.model.vae.encode(rgb).latent_dist.mode()*self.model.vae.config.scaling_factor)
            latents=torch.cat(initial).unsqueeze(0).permute(0,2,1,3,4)
            count=max(1,int(steps*denoise_strength))
            timesteps=timesteps[-count:]
            latents=scheduler.add_noise(latents,noise,timesteps[:1])
        embedding=self.model.text_embeddings(1,self.model.config.prompt)
        for step,t in enumerate(timesteps):
            prediction=torch.zeros_like(latents,dtype=torch.float32)
            weights=torch.zeros((1,1,frames,1,1),device=self.device,dtype=torch.float32)
            model_input=scheduler.scale_model_input(latents,t)
            for indices in windows:
                sample=model_input[:,:,indices]
                flat=sample.permute(0,2,1,3,4).reshape(len(indices),4,*sample.shape[-2:])
                text=embedding.repeat_interleave(len(indices),dim=0)
                down,mid=self.model.controlnet(flat,t,encoder_hidden_states=text,
                    controlnet_cond=encoded[indices],conditioning_scale=1.,return_dict=False)
                value=self.motion(sample,t,encoder_hidden_states=text,
                    down_block_additional_residuals=down,mid_block_additional_residual=mid).sample
                n=len(indices)
                w=torch.minimum(torch.arange(1,n+1,device=self.device),
                                torch.arange(n,0,-1,device=self.device)).float().view(1,1,n,1,1)
                prediction[:,:,indices]+=value.float()*w
                weights[:,:,indices]+=w
            if not torch.all(weights>0):raise AssertionError('Uncovered temporal frame')
            predicted=(prediction/weights).to(self.dtype)
            if scheduler_kind=='ddpm-shared':
                latents=shared_ddpm_step(scheduler,predicted,t,latents,generator)
            else:
                latents=scheduler.step(predicted,t,latents,generator=generator).prev_sample
            if not torch.isfinite(latents).all():raise RuntimeError('Nonfinite temporal output')
            if progress is not None:progress(step+1,len(timesteps))
        flat=latents.permute(0,2,1,3,4).reshape(frames,4,*latents.shape[-2:])
        result=[]
        for start in range(0,frames,2):
            decoded=self.model.vae.decode(flat[start:start+2]/self.model.vae.config.scaling_factor).sample
            rgb=((decoded.float().clamp(-1,1)+1)*127.5).round().to(torch.uint8)
            result.extend(rgb.permute(0,2,3,1).cpu().numpy())
        return np.stack(result)
