# importing libraries
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSMBlock(nn.Module):
    """PyTorch-native selective state-space block."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.dt_rank = max(1, (d_model + 15) // 16)

        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=3,
            groups=self.d_inner,
            padding=0,
        )

        self.x_proj = nn.Linear(
            self.d_inner,
            self.dt_rank + 2 * d_state,
            bias=False,
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        state_values = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(state_values.log().repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)

        u, z = self.in_proj(x).chunk(2, dim=-1)

        u = F.pad(u.transpose(1, 2), (2, 0))
        u = self.conv1d(u).transpose(1, 2)
        u = F.silu(u)

        projected = self.x_proj(u)
        dt_low, b_values, c_values = torch.split(
            projected,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )

        dt = F.softplus(self.dt_proj(dt_low))
        A = -torch.exp(self.A_log).to(dtype=u.dtype)
        D = self.D.to(dtype=u.dtype)

        batch_size, seq_len, _ = u.shape
        state = u.new_zeros(
            batch_size,
            self.d_inner,
            self.d_state,
        )

        outputs = []

        for t in range(seq_len):
            dt_t = dt[:, t, :]
            b_t = b_values[:, t, :]
            c_t = c_values[:, t, :]
            u_t = u[:, t, :]

            transition = torch.exp(
                dt_t.unsqueeze(-1) * A.unsqueeze(0)
            )

            input_update = (
                dt_t.unsqueeze(-1)
                * b_t.unsqueeze(1)
                * u_t.unsqueeze(-1)
            )

            state = transition * state + input_update

            y_t = (
                state * c_t.unsqueeze(1)
            ).sum(dim=-1) + D * u_t

            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)
        y = y * F.silu(z)

        return residual + self.dropout(self.out_proj(y))


class ECGMamba(nn.Module):
    """
    Mamba-inspired selective state-space model for multi-label ECG.

    Input:  (batch, 12, 1000)
    Output: (batch, n_labels), raw logits.
    """

    def __init__(
        self,
        n_labels: int,
        in_channels: int = 12,
        d_model: int = 64,
        d_state: int = 16,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(
                in_channels,
                d_model,
                kernel_size=15,
                stride=4,
                padding=7,
                bias=False,
            ),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(
                d_model,
                d_model,
                kernel_size=5,
                stride=2,
                padding=2,
                bias=False,
            ),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )

        self.blocks = nn.ModuleList(
            [
                SelectiveSSMBlock(
                    d_model=d_model,
                    d_state=d_state,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, n_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = x.transpose(1, 2)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = x.mean(dim=1)

        return self.classifier(x)


if __name__ == "__main__":
    model = ECGMamba(n_labels=44)
    dummy = torch.randn(4, 12, 1000)
    output = model(dummy)

    print("Input shape:", tuple(dummy.shape))
    print("Output shape:", tuple(output.shape))
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
