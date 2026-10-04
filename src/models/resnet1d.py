"""
1D ResNet for multi-label ECG classification.
Sized for fast training on an 8GB GPU — this is a lightweight architecture;
the bottleneck for this project is never model size, it's the trustworthiness
analysis built on top of it.
"""
import torch
import torch.nn as nn


class ResBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet1D(nn.Module):
    """
    Input:  (batch, 12, 1000)  — 12-lead ECG at 100Hz, 10 seconds
    Output: (batch, n_labels)  — raw logits (apply sigmoid for probabilities,
                                  multi-label so NOT softmax)
    """

    def __init__(self, n_labels: int, in_channels: int = 12, base_width: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_width, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base_width),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = ResBlock1D(base_width, base_width, stride=1)
        self.layer2 = ResBlock1D(base_width, base_width * 2, stride=2)
        self.layer3 = ResBlock1D(base_width * 2, base_width * 4, stride=2)
        self.layer4 = ResBlock1D(base_width * 4, base_width * 8, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_width * 8, n_labels)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


if __name__ == "__main__":
    model = ResNet1D(n_labels=40)
    dummy = torch.randn(4, 12, 1000)
    out = model(dummy)
    print("Output shape:", out.shape)  # expect (4, 40)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
