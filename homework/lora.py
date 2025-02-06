from pathlib import Path

import torch

from .bignet import BIGNET_DIM, LayerNorm  # noqa: F401
from .half_precision import HalfLinear


class LoRALinear(HalfLinear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_dim: int,
        alpha: float = 16.0,
        bias: bool = True,
    ) -> None:
        """
        Implement the LoRALinear layer as described in the homework

        Hint: You can use the HalfLinear class as a parent class (it makes load_state_dict easier, names match)
        Hint: Remember to initialize the weights of the lora layers
        Hint: Make sure the linear layers are not trainable, but the LoRA layers are
        """
        super().__init__(in_features, out_features, bias)
        
        #freeze the model params
        for param in self.parameters():
            param.requires_grad=False

        #initialize the lora layers
        self.lora_a = torch.nn.Linear(in_features, lora_dim, bias=False, dtype = torch.float32)
        self.lora_b = torch.nn.Linear(lora_dim, out_features, bias=False, dtype = torch.float32)

        self.alpha_div_rank = alpha / lora_dim

        torch.nn.init.kaiming_uniform_(self.lora_a.weight)
        torch.nn.init.zeros_(self.lora_b.weight)

        self.lora_a.requires_grad_(True)
        self.lora_b.requires_grad_(True)


        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = super().forward(x.to(torch.float16))  # Base model uses float16

        # Compute LoRA output (LoRA layers work in float32)
        lora_out = self.alpha_div_rank * self.lora_b(self.lora_a(x.to(torch.float32)))

        # Cast LoRA output to base_out dtype and sum
        return base_out + lora_out.to(base_out.dtype)


class LoraBigNet(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, channels: int, lora_dim: int):
            super().__init__()

            #implement the block using loRAlinear
            self.model = torch.nn.Sequential(
                LoRALinear(channels, channels, lora_dim),
                torch.nn.ReLU(),
                LoRALinear(channels, channels, lora_dim),
                torch.nn.ReLU(),
                LoRALinear(channels, channels, lora_dim)
            )


        def forward(self, x: torch.Tensor):
            return self.model(x) + x

    def __init__(self, lora_dim: int = 32):
        super().__init__()
        #making sure to structure the model as described
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM, lora_dim),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None) -> LoraBigNet:
    # Since we have additional layers, we need to set strict=False in load_state_dict
    net = LoraBigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True), strict=False)
    return net
