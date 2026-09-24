"""Checkpoint-compatible Ours path, extracted from the research drafter.

Feature selections are implementation choices, not attributed to another paper.
Only full-memory cross attention and the learned Gaussian-centre path remain.
"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


@dataclass
class DraftConfig:
    hidden_size: int = 1024
    attention_heads: int = 16
    ffn_hidden: int = 2048
    feature_layers: tuple = (0, 9, 18, 27)
    feature_noise: float = 0.6


class Drafter(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or DraftConfig()
        h = self.config.hidden_size
        self.restart_fusion = nn.Linear(5 * h, h)
        self.self_norm = nn.LayerNorm(h)
        self.self_qkv = nn.Linear(h, 3 * h)
        self.self_out = nn.Linear(h, h)
        self.cross_norm = nn.LayerNorm(h)
        self.cross_q = nn.Linear(h, h)
        self.cross_kv = nn.Linear(h, 2 * h)
        self.cross_out = nn.Linear(h, h)
        self.ffn_norm = nn.LayerNorm(h)
        self.ffn = nn.Sequential(nn.Linear(h, self.config.ffn_hidden), nn.GELU(), nn.Linear(self.config.ffn_hidden, h))
        self.output_norm = nn.LayerNorm(h)

    def _heads(self, x):
        return x.unflatten(-1, (self.config.attention_heads, -1)).transpose(1, 2)

    def forward(self, token_embedding, audio_memory, audio_mask, *,
                target_features=None, recurrent_state=None, past_kv=None,
                audio_time_positions=None, acoustic_sigma_seconds=None,
                acoustic_anchor_adapter=None):
        if recurrent_state is None:
            assert target_features is not None and past_kv is None and len(target_features) == 4
            features = target_features
            if self.training and self.config.feature_noise:
                features = [f + torch.randn_like(f) * self.config.feature_noise for f in features]
            x = self.restart_fusion(torch.cat([token_embedding, *features], dim=-1))
        else:
            assert target_features is None and past_kv is not None
            x = recurrent_state + token_embedding
        q, k, v = [self._heads(t) for t in self.self_qkv(self.self_norm(x)).chunk(3, dim=-1)]
        assert x.shape[1] == 1
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        a = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        x = x + self.self_out(a.transpose(1, 2).flatten(2))
        center = acoustic_anchor_adapter(x) if acoustic_anchor_adapter is not None else None
        q = self._heads(self.cross_q(self.cross_norm(x)))
        ak, av = [self._heads(t) for t in self.cross_kv(audio_memory).chunk(2, dim=-1)]
        if center is None:
            assert audio_time_positions is None and acoustic_sigma_seconds is None
            mask = audio_mask[:, None, None, :]
        else:
            assert audio_time_positions.shape == audio_mask.shape and acoustic_sigma_seconds > 0
            bias = -((audio_time_positions - center[:, None]) ** 2) / (2 * acoustic_sigma_seconds ** 2)
            mask = bias.to(q.dtype).masked_fill(~audio_mask, float('-inf'))[:, None, None, :]
        if center is not None and center.requires_grad:
            # Preserve the original gradient-capable backend for additive masks.
            with sdpa_kernel(SDPBackend.MATH):
                a = F.scaled_dot_product_attention(q, ak, av, attn_mask=mask.contiguous(), dropout_p=0.0)
        else:
            a = F.scaled_dot_product_attention(q, ak, av, attn_mask=mask, dropout_p=0.0)
        x = x + self.cross_out(a.transpose(1, 2).flatten(2))
        z = self.output_norm(x + self.ffn(self.ffn_norm(x)))
        return z, (k, v)
