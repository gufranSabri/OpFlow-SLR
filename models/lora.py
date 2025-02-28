import torch
from torch import nn

class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha=1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha  # Scaling factor
        self.A = nn.Parameter(torch.randn(in_dim, rank) * 0.01)
        self.B = nn.Parameter(torch.randn(rank, out_dim) * 0.01)

    def forward(self, x):
        return self.alpha * (x @ self.A @ self.B)

class MultiRankLoRA(nn.Module):
    """Applies multiple LoRA modules in parallel with different ranks."""
    def __init__(self, in_dim, out_dim, ranks, alphas):
        super().__init__()
        assert len(ranks) == len(alphas), "Each rank must have a corresponding alpha."
        self.lora_layers = nn.ModuleList([LoRALayer(in_dim, out_dim, r, a) for r, a in zip(ranks, alphas)])

    def forward(self, x):
        return sum(lora(x) for lora in self.lora_layers)
    

if __name__ == "__main__":
    lora = MultiRankLoRA(10, 10, [4, 8, 16], [1.0, 2.0, 4.0])

    x = torch.randn(10, 10)
    print(lora(x).shape)