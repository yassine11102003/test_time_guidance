"""Stable Diffusion 1.5: loading, prompt encoding, CFG noise prediction, differentiable VAE decoding."""
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
LATENT_SCALE = 0.18215  # SD1.5 VAE scaling factor


class StableDiffusion:
    def __init__(self, device="cuda", unet_dtype=torch.bfloat16):
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID, torch_dtype=unet_dtype, safety_checker=None, requires_safety_checker=False
        )
        # Default scheduler is PNDM; we only need DDIM's alphas (our own loop does the steps).
        self.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        self.pipe = pipe  # kept for the diffusers reference check (scripts/check_setup.py)

        self.device, self.unet_dtype = device, unet_dtype
        self.unet = pipe.unet.to(device)
        self.text_encoder = pipe.text_encoder.to(device)
        self.tokenizer = pipe.tokenizer
        # The VAE stays in fp32: every reward gradient flows back through it.
        self.vae = pipe.vae.to(device, torch.float32)
        for m in (self.unet, self.text_encoder, self.vae):
            m.requires_grad_(False)
            m.eval()

        self.alphas_cumprod = self.scheduler.alphas_cumprod.double()  # \bar\alpha_t, t = 0..999
        self.final_alpha_cumprod = float(self.scheduler.final_alpha_cumprod)  # = \bar\alpha_0 for SD

    @torch.no_grad()
    def encode_prompt(self, prompt):
        """Returns (uncond, cond) text embeddings, each [1, 77, 768]."""
        def enc(text):
            ids = self.tokenizer(
                text, padding="max_length", max_length=self.tokenizer.model_max_length,
                truncation=True, return_tensors="pt",
            ).input_ids.to(self.device)
            return self.text_encoder(ids)[0]
        return enc(""), enc(prompt)

    def eps(self, x, t, emb, cfg=7.5):
        """Classifier-free guidance noise prediction (Ho & Salimans, 2022). Differentiable w.r.t. x when grad is enabled."""
        uncond, cond = emb
        b = x.shape[0]
        x_in = torch.cat([x, x]).to(self.unet_dtype)
        e_in = torch.cat([uncond.expand(b, -1, -1), cond.expand(b, -1, -1)])
        out = self.unet(x_in, t, encoder_hidden_states=e_in).sample.float()
        e_u, e_c = out.chunk(2)
        return e_u + cfg * (e_c - e_u)

    def decode(self, x0):
        """Latent -> image y in [0, 1]^{3x512x512}. Differentiable (clamp zeroes grads of saturated pixels)."""
        y = self.vae.decode(x0.float() / LATENT_SCALE).sample
        return (y / 2 + 0.5).clamp(0, 1)

    def bind(self, prompt, reward=None, cfg=7.5):
        """Model interface used by the sampler and the guidance methods, for one prompt."""
        return PromptModel(self, prompt, reward, cfg)


class PromptModel:
    """What tg.ddim.sample and the guidance methods see: eps(x, t) and reward(x0)."""

    def __init__(self, sd, prompt, reward, cfg):
        self.sd, self.prompt, self.reward_fn, self.cfg = sd, prompt, reward, cfg
        self.emb = sd.encode_prompt(prompt)
        self.alphas_cumprod = sd.alphas_cumprod
        self.final_alpha_cumprod = sd.final_alpha_cumprod
        self.n_unet = 0  # number of UNet evaluations (per image, CFG pair counted once)

    def eps(self, x, t):
        self.n_unet += x.shape[0]
        return self.sd.eps(x, t, self.emb, self.cfg)

    def reward(self, x0):
        """R(x0) := R(D(x0)), shape [B]."""
        return self.reward_fn(self.sd.decode(x0), self.prompt)
