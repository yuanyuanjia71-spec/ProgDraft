"""Shared nonnegative acoustic displacement predictor."""
import torch
from torch import nn
from torch.nn import functional as F
HIDDEN = 1024
SIGMA_SECONDS = 0.2

class AcousticProgressPredictor(nn.Module):
    """The requested shared A2 predictor: [h_pre, a/duration] -> Softplus delta."""

    def __init__(self, hidden_size: int = HIDDEN) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.network = nn.Sequential(
            nn.Linear(self.hidden_size + 1, 512), nn.GELU(),
            nn.Linear(512, 128), nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, hidden_pre: torch.Tensor, current_position_s: torch.Tensor,
                audio_duration_s: torch.Tensor) -> torch.Tensor:
        if hidden_pre.ndim != 3 or tuple(hidden_pre.shape[1:]) != (1, self.hidden_size):
            raise ValueError(f'hidden_pre must be [batch, 1, {self.hidden_size}]')
        batch = hidden_pre.shape[0]
        if current_position_s.shape != (batch,) or audio_duration_s.shape != (batch,):
            raise ValueError('position and duration must be [batch]')
        if not bool((audio_duration_s > 0).all()):
            raise ValueError('audio durations must be positive')
        inputs = torch.cat((hidden_pre[:, 0].float(),
                            (current_position_s / audio_duration_s).float().unsqueeze(-1)), dim=-1)
        return F.softplus(self.network(inputs).squeeze(-1))

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def _center(position_s: torch.Tensor, duration_s: torch.Tensor) -> torch.Tensor:
    """Clip only the Gaussian centre, preserving raw recursive state/gradients."""
    return torch.minimum(torch.maximum(position_s, torch.zeros_like(position_s)), duration_s)
