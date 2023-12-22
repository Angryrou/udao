import torch
import torch.nn as nn


class IsoBN(nn.Module):
    """Isotropic Batch Normalization."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.cov = nn.Parameter(torch.zeros(hidden_dim, hidden_dim))
        self.std = nn.Parameter(torch.zeros(hidden_dim))

    def forward(
        self,
        input: torch.Tensor,
        momentum: float = 0.05,
        eps: float = 1e-3,
        beta: float = 0.5,
    ) -> torch.Tensor:
        with torch.no_grad():
            if self.training:
                x = input.detach()
                n = x.size(0)
                mean = x.mean(dim=0)
                y = x - mean.unsqueeze(0)
                std = (y**2).mean(0) ** 0.5
                cov = (y.t() @ y) / n
                self.cov.data += momentum * (cov.data - self.cov.data)
                self.std.data += momentum * (std.data - self.std.data)

            corr = torch.clamp(
                self.cov
                / (
                    torch.ger(self.std, self.std)
                    + torch.full_like(torch.ger(self.std, self.std), 1e-6)
                ),
                -1,
                1,
            )
            gamma = (corr**2).mean(1)
            denorm = gamma * self.std
            scale = 1 / (denorm + eps) ** beta
            E = torch.diag(self.cov).sum()
            new_E = (torch.diag(self.cov) * (scale**2)).sum()
            m = (E / (new_E + eps)) ** 0.5
            scale *= m
        return input * scale.unsqueeze(0).detach()
