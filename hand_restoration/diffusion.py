from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


LOSS_STAT_NAMES = (
    "weighted_error_sum",
    "weighted_count",
    "unweighted_error_sum",
    "unweighted_count",
    "hand_error_sum",
    "hand_count",
    "background_error_sum",
    "background_count",
)


def regional_weighted_mse(
    squared_error: torch.Tensor,
    edit_mask: torch.Tensor | None,
    hand_loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the weighted MSE and detached sums/counts for regional logging."""
    if hand_loss_weight < 1.0:
        raise ValueError("hand_loss_weight must be at least 1.0")
    if squared_error.ndim != 4:
        raise ValueError("squared_error must be a BCHW tensor")

    if edit_mask is None:
        latent_mask = torch.zeros_like(squared_error[:, :1], dtype=torch.float32)
    else:
        mask = edit_mask.to(
            device=squared_error.device, dtype=squared_error.dtype, non_blocking=True
        ).clamp(0.0, 1.0)
        # Max pooling keeps thin fingers represented after the 512 -> 64 latent
        # reduction.
        latent_mask = F.adaptive_max_pool2d(mask, squared_error.shape[-2:]).float()

    channels = squared_error.shape[1]
    weights = 1.0 + (float(hand_loss_weight) - 1.0) * latent_mask
    weighted_error_sum = (squared_error * weights).sum()
    weighted_count = weights.sum() * channels
    unweighted_error_sum = squared_error.sum()
    unweighted_count = squared_error.new_tensor(squared_error.numel())
    hand_error_sum = (squared_error * latent_mask).sum()
    hand_count = latent_mask.sum() * channels
    background_mask = 1.0 - latent_mask
    background_error_sum = (squared_error * background_mask).sum()
    background_count = background_mask.sum() * channels

    statistics = torch.stack(
        (
            weighted_error_sum,
            weighted_count,
            unweighted_error_sum,
            unweighted_count,
            hand_error_sum,
            hand_count,
            background_error_sum,
            background_count,
        )
    ).detach()
    return weighted_error_sum / weighted_count.clamp_min(1.0), statistics


def _require_diffusers() -> None:
    try:
        import accelerate  # noqa: F401
        import diffusers  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Diffusion dependencies are missing. Install the glove-hot3d additions "
            "from environment.glove-hot3d.yml, then rerun this command."
        ) from exc


@dataclass(frozen=True)
class DiffusionConfig:
    base_model_id: str = "runwayml/stable-diffusion-v1-5"
    controlnet_model_id: str | None = None
    prompt: str = "a realistic egocentric image of a human hand"
    negative_prompt: str = "deformed hand, extra fingers, blurry, cartoon"
    prediction_type: str | None = None


class ControlNetHandRestorer(torch.nn.Module):
    """Frozen SD 1.5 backbone with a trainable ControlNet condition branch."""

    def __init__(self, config: DiffusionConfig, device: str | torch.device = "cpu") -> None:
        super().__init__()
        _require_diffusers()
        from diffusers import ControlNetModel, DDPMScheduler, UNet2DConditionModel, AutoencoderKL
        from transformers import CLIPTextModel, CLIPTokenizer

        self.config = config
        self.device_name = torch.device(device)
        self.tokenizer = CLIPTokenizer.from_pretrained(config.base_model_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(config.base_model_id, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(config.base_model_id, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(config.base_model_id, subfolder="unet")
        self.noise_scheduler = DDPMScheduler.from_pretrained(config.base_model_id, subfolder="scheduler")
        if config.controlnet_model_id:
            self.controlnet = ControlNetModel.from_pretrained(config.controlnet_model_id)
        else:
            self.controlnet = ControlNetModel.from_unet(self.unet)
        for module in (self.vae, self.text_encoder, self.unet):
            module.requires_grad_(False)
            module.eval()
        if config.prediction_type:
            self.noise_scheduler.register_to_config(prediction_type=config.prediction_type)

    @property
    def trainable_parameters(self):
        return self.controlnet.parameters()

    @torch.no_grad()
    def text_embeddings(self, batch_size: int, prompt: str | None = None) -> torch.Tensor:
        tokens = self.tokenizer(
            [prompt or self.config.prompt] * batch_size,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device_name)
        return self.text_encoder(tokens)[0]

    def training_loss(
        self,
        target_rgb: torch.Tensor,
        condition_rgb: torch.Tensor,
        edit_mask: torch.Tensor | None = None,
        hand_loss_weight: float = 1.0,
        generator: torch.Generator | None = None,
        return_statistics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Latent diffusion objective with optional edit-region weighting.

        Both inputs must be Bx3xHxW float images in [-1, 1].
        """
        target_rgb = target_rgb.to(self.device_name)
        condition_rgb = condition_rgb.to(self.device_name)
        with torch.no_grad():
            latents = self.vae.encode(target_rgb).latent_dist.sample(generator=generator) * self.vae.config.scaling_factor
            text = self.text_embeddings(target_rgb.shape[0])
        noise = torch.randn(latents.shape, generator=generator, device=latents.device, dtype=latents.dtype)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (latents.shape[0],),
            generator=generator, device=latents.device, dtype=torch.long,
        )
        noisy = self.noise_scheduler.add_noise(latents, noise, timesteps)
        down, mid = self.controlnet(
            noisy,
            timesteps,
            encoder_hidden_states=text,
            controlnet_cond=(condition_rgb + 1.0) / 2.0,
            return_dict=False,
        )
        prediction = self.unet(
            noisy,
            timesteps,
            encoder_hidden_states=text,
            down_block_additional_residuals=down,
            mid_block_additional_residual=mid,
        ).sample
        target = noise
        if self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        squared_error = (prediction.float() - target.float()).square()
        loss, statistics = regional_weighted_mse(
            squared_error, edit_mask, hand_loss_weight
        )
        if return_statistics:
            return loss, statistics
        return loss

    def save_controlnet(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": self.config.__dict__, "state_dict": self.controlnet.state_dict()}, path)

    def load_controlnet(self, path: str | Path, strict: bool = True) -> None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.controlnet.load_state_dict(checkpoint["state_dict"], strict=strict)

    @torch.no_grad()
    def generate(self, condition_rgb: torch.Tensor, steps: int = 30, guidance_scale: float = 5.0, controlnet_scale: float = 1.0, seed: int = 0):
        from diffusers import StableDiffusionControlNetPipeline

        condition = condition_rgb.to(self.device_name)
        pipeline = StableDiffusionControlNetPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            controlnet=self.controlnet,
            scheduler=self.noise_scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        ).to(self.device_name)
        generator = torch.Generator(device=self.device_name).manual_seed(seed)
        from PIL import Image
        image = ((condition[0].detach().cpu().permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype("uint8")
        return pipeline(
            prompt=self.config.prompt,
            negative_prompt=self.config.negative_prompt,
            image=Image.fromarray(image),
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_scale,
            generator=generator,
        ).images[0]
