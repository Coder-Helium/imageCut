from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch import nn


@dataclass
class BackboneOutput:
    tokens: torch.Tensor
    pooled: torch.Tensor
    spatial_size: Tuple[int, int]


class CompactViTBackbone(nn.Module):
    """Small local ViT-style fallback used for smoke tests.

    Production configs should use a foundation backbone such as DINOv3 through
    Hugging Face, torch.hub, or timm. This fallback keeps the RIGFormer code
    executable in clean environments without downloading weights.
    """

    def __init__(
        self,
        output_dim: int = 256,
        width: int = 64,
        patch_size: int = 16,
        depth: int = 2,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, width, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(width),
            nn.GELU(),
            nn.Conv2d(width, output_dim, 3, stride=max(1, patch_size // 2), padding=1, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.GELU(),
        )
        self.pos_proj = nn.Linear(2, output_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=max(1, num_heads),
            dim_feedforward=output_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=max(1, depth))
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, image: torch.Tensor) -> BackboneOutput:
        feat = self.patch_embed(image)
        b, c, h, w = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)
        tokens = tokens + _coord_positional_tokens(h, w, image.device, image.dtype, self.pos_proj)
        tokens = self.norm(self.encoder(tokens))
        return BackboneOutput(tokens=tokens, pooled=tokens.mean(dim=1), spatial_size=(h, w))


class TorchvisionConvNeXtBackbone(nn.Module):
    """Torchvision ConvNeXt wrapper for environments without timm."""

    def __init__(self, output_dim: int = 256, name: str = "convnext_tiny", pretrained: bool = False) -> None:
        super().__init__()
        try:
            import torchvision.models as tvm
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("torchvision is required for torchvision ConvNeXt backbones") from exc
        if not hasattr(tvm, name):
            raise ValueError(f"torchvision.models has no backbone named {name}")
        weights = "DEFAULT" if pretrained else None
        model = getattr(tvm, name)(weights=weights)
        self.features = model.features
        self.project = nn.LazyConv2d(output_dim, kernel_size=1)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, image: torch.Tensor) -> BackboneOutput:
        feat = self.features(image)
        feat = self.project(feat)
        b, c, h, w = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)
        tokens = self.norm(tokens)
        return BackboneOutput(tokens=tokens, pooled=tokens.mean(dim=1), spatial_size=(h, w))


class HuggingFaceBackbone(nn.Module):
    """Optional Hugging Face wrapper for DINOv3 / DINOv2 / CLIP-style encoders."""

    def __init__(
        self,
        output_dim: int = 256,
        name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        pretrained: bool = True,
        trust_remote_code: bool = True,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoConfig, AutoModel
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("transformers is required for Hugging Face backbones") from exc
        if pretrained:
            self.model = AutoModel.from_pretrained(name, trust_remote_code=trust_remote_code)
            hidden = int(getattr(self.model.config, "hidden_size", output_dim))
        else:
            cfg = AutoConfig.from_pretrained(name, trust_remote_code=trust_remote_code)
            self.model = AutoModel.from_config(cfg, trust_remote_code=trust_remote_code)
            hidden = int(getattr(cfg, "hidden_size", output_dim))
        self.project = nn.Identity() if hidden == output_dim else nn.Linear(hidden, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, image: torch.Tensor) -> BackboneOutput:
        out = self.model(pixel_values=image)
        tokens = getattr(out, "last_hidden_state", None)
        if tokens is None:
            raise RuntimeError("Hugging Face backbone did not return last_hidden_state")
        if tokens.size(1) > 1:
            pooled = tokens[:, 0]
            tokens = tokens[:, 1:]
        else:
            pooled = tokens.mean(dim=1)
        tokens = self.norm(self.project(tokens))
        pooled = self.norm(self.project(pooled))
        side = int(tokens.size(1) ** 0.5)
        return BackboneOutput(tokens=tokens, pooled=pooled, spatial_size=(side, side))


class TimmBackbone(nn.Module):
    """Optional timm wrapper for DINOv3 and other current vision backbones."""

    def __init__(
        self,
        output_dim: int = 256,
        name: str = "vit_base_patch16_dinov3.lvd1689m",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        try:
            import timm
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("timm is required for timm backbones") from exc
        self.model = timm.create_model(name, pretrained=pretrained, num_classes=0)
        hidden = int(getattr(self.model, "num_features", output_dim))
        self.project = nn.Identity() if hidden == output_dim else nn.Linear(hidden, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, image: torch.Tensor) -> BackboneOutput:
        if hasattr(self.model, "forward_features"):
            feat = self.model.forward_features(image)
        else:
            feat = self.model(image)
        if feat.dim() == 4:
            if feat.shape[1] < feat.shape[-1]:
                b, c, h, w = feat.shape
                tokens = feat.flatten(2).transpose(1, 2)
            else:
                b, h, w, c = feat.shape
                tokens = feat.reshape(b, h * w, c)
        elif feat.dim() == 3:
            tokens = feat[:, 1:] if feat.size(1) > 1 else feat
            h = w = int(tokens.size(1) ** 0.5)
        elif feat.dim() == 2:
            tokens = feat.unsqueeze(1)
            h = w = 1
        else:
            raise RuntimeError(f"Unsupported timm feature shape: {tuple(feat.shape)}")
        tokens = self.norm(self.project(tokens))
        return BackboneOutput(tokens=tokens, pooled=tokens.mean(dim=1), spatial_size=(h, w))


class TorchHubDINOv3Backbone(nn.Module):
    """Optional Meta torch.hub DINOv3 wrapper.

    The exact hub model name is kept configurable because DINOv3 checkpoints
    differ by size and release channel.
    """

    def __init__(
        self,
        output_dim: int = 256,
        repo: str = "facebookresearch/dinov3",
        name: str = "dinov3_vitb16",
        pretrained: bool = True,
        weights: str | None = None,
        source: str | None = None,
        trust_repo: bool | str | None = None,
        force_reload: bool = False,
        check_hash: bool = False,
    ) -> None:
        super().__init__()
        hub_kwargs: Dict[str, Any] = {
            "pretrained": bool(pretrained or weights),
            "force_reload": force_reload,
        }
        if source is None:
            source = "local" if Path(repo).expanduser().exists() else "github"
        if trust_repo is not None:
            hub_kwargs["trust_repo"] = trust_repo
        if weights:
            weights_path = Path(weights).expanduser()
            hub_kwargs["weights"] = str(weights_path if not weights_path.exists() else weights_path.resolve())
        if check_hash:
            hub_kwargs["check_hash"] = True
        self.model = torch.hub.load(repo, name, source=source, **hub_kwargs)
        hidden = int(getattr(self.model, "embed_dim", getattr(self.model, "num_features", output_dim)))
        self.project = nn.Identity() if hidden == output_dim else nn.Linear(hidden, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, image: torch.Tensor) -> BackboneOutput:
        if hasattr(self.model, "forward_features"):
            feat = self.model.forward_features(image)
            if isinstance(feat, dict):
                tokens = feat.get("x_norm_patchtokens")
                if tokens is None:
                    tokens = feat.get("patch_tokens")
                if tokens is None:
                    tokens = feat.get("tokens")
                cls = feat.get("x_norm_clstoken")
                if cls is None:
                    cls = feat.get("cls_token")
                if tokens is None:
                    raise RuntimeError("DINOv3 hub model did not expose patch tokens")
                pooled = cls if cls is not None else tokens.mean(dim=1)
            else:
                tokens = feat[:, 1:] if feat.dim() == 3 and feat.size(1) > 1 else feat
                pooled = tokens.mean(dim=1)
        else:
            pooled = self.model(image)
            tokens = pooled.unsqueeze(1)
        tokens = self.norm(self.project(tokens))
        pooled = self.norm(self.project(pooled))
        side = int(tokens.size(1) ** 0.5)
        return BackboneOutput(tokens=tokens, pooled=pooled, spatial_size=(side, side))


def build_visual_backbone(config: Dict[str, Any] | None, output_dim: int, fallback_width: int = 64) -> nn.Module:
    cfg = dict(config or {})
    kind = str(cfg.pop("type", cfg.pop("kind", "compact_vit"))).lower()
    freeze = bool(cfg.pop("freeze", False))
    if kind in {"compact", "compact_vit", "fallback", "smoke"}:
        model = CompactViTBackbone(output_dim=output_dim, width=int(cfg.pop("width", fallback_width)), **cfg)
    elif kind in {"hf", "huggingface", "dinov3_hf", "dinov2_hf", "clip_hf"}:
        model = HuggingFaceBackbone(output_dim=output_dim, **cfg)
    elif kind in {"timm", "dinov3_timm", "swin_timm", "convnext_timm"}:
        model = TimmBackbone(output_dim=output_dim, **cfg)
    elif kind in {"torchhub_dinov3", "dinov3_torchhub"}:
        model = TorchHubDINOv3Backbone(output_dim=output_dim, **cfg)
    elif kind in {"torchvision_convnext", "convnext_torchvision"}:
        model = TorchvisionConvNeXtBackbone(output_dim=output_dim, **cfg)
    else:
        raise ValueError(f"Unknown RIGFormer backbone type: {kind}")
    if freeze:
        for param in model.parameters():
            param.requires_grad_(False)
    return model


def _coord_positional_tokens(
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    proj: nn.Linear,
) -> torch.Tensor:
    ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([xx, yy], dim=-1).reshape(1, height * width, 2)
    return proj(coords)
