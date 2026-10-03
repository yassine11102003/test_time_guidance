"""Differentiable rewards on images y in [0, 1]^{B x 3 x H x W}. Each returns a tensor of shape [B]."""
import torch
import torch.nn.functional as F

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def luminance_map(y):
    """Per-pixel luminance l(y) in [0, 1]^{B x H x W}."""
    return 0.299 * y[:, 0] + 0.587 * y[:, 1] + 0.114 * y[:, 2]


class LuminanceReward:
    """R_lum(y) = -(mean luminance - target)^2."""

    def __init__(self, target=0.2):
        self.target = target

    def __call__(self, y, prompt=None):
        return -(luminance_map(y).mean(dim=(1, 2)) - self.target) ** 2


class ImageReward:
    """Torch re-implementation of ImageReward.score (which goes through PIL and is not differentiable)."""

    def __init__(self, device="cuda"):
        import ImageReward as RM
        self.model = RM.load("ImageReward-v1.0", device=device).eval()
        self.model.requires_grad_(False)
        self.device = device

    def __call__(self, y, prompt):
        m, b = self.model, y.shape[0]
        x = F.interpolate(y, size=(224, 224), mode="bicubic", align_corners=False, antialias=True)
        x = (x - CLIP_MEAN.to(x)) / CLIP_STD.to(x)
        tok = m.blip.tokenizer(prompt, padding="max_length", truncation=True,
                               max_length=35, return_tensors="pt").to(self.device)
        img = m.blip.visual_encoder(x)
        img_att = torch.ones(img.shape[:-1], dtype=torch.long, device=self.device)
        out = m.blip.text_encoder(
            tok.input_ids.expand(b, -1), attention_mask=tok.attention_mask.expand(b, -1),
            encoder_hidden_states=img, encoder_attention_mask=img_att, return_dict=True,
        )
        r = m.mlp(out.last_hidden_state[:, 0, :].float()).squeeze(-1)
        return (r - m.mean) / m.std


def make_reward(name, device="cuda", lum_target=0.2):
    if name == "luminance":
        return LuminanceReward(lum_target)
    if name == "imagereward":
        return ImageReward(device)
    raise ValueError(f"unknown reward: {name}")
