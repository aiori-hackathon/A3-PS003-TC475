"""
PyTorch Compact GRU Model for Traffic Flow Forecasting (PS-003).
Lightweight recurrent architecture optimized for CPU and laptop execution.
"""

from typing import Dict
import copy
import torch
import torch.nn as nn


class TrafficGRU(nn.Module):
    """
    Compact GRU network for single-step traffic flow forecasting.
    Input shape: (batch_size, window_length, 1)
    Output shape: (batch_size, 1)
    """
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 32,
        num_layers: int = 1,
        output_size: int = 1,
        dropout: float = 0.0
    ):
        super(TrafficGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, input_size)
        out, _ = self.gru(x)
        # Take the output of the last time step
        last_step = out[:, -1, :]
        prediction = self.fc(last_step)
        return prediction

    def count_parameters(self) -> int:
        """Returns the exact count of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_architecture_summary(self, window_length: int = 12) -> dict:
        """Returns architectural metadata for dashboard display."""
        total_params = self.count_parameters()
        return {
            "model_type": "Gated Recurrent Unit (GRU)",
            "input_window": window_length,
            "input_features": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "output_size": self.output_size,
            "total_parameters": total_params,
            "param_breakdown": {
                name: param.numel() for name, param in self.named_parameters()
            }
        }


def get_weights_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extracts a detached CPU copy of state_dict."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def set_weights_dict(model: nn.Module, weights_dict: Dict[str, torch.Tensor]):
    """Loads weights into model."""
    model.load_state_dict(weights_dict)


def clone_model(model: nn.Module) -> nn.Module:
    """Creates a deep copy of the model."""
    return copy.deepcopy(model)


def quantize_to_fp16(weights_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Simulates FP16 transmission: casts Float32 parameters to Half (Float16)
    and casts back to Float32 at receiver.
    """
    fp16_dict = {}
    for k, v in weights_dict.items():
        if v.is_floating_point():
            half_tensor = v.to(torch.float16)
            fp16_dict[k] = half_tensor.to(torch.float32)
        else:
            fp16_dict[k] = v.clone()
    return fp16_dict
