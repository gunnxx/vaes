import torch
import torch.nn as nn

class ResidualLayer(nn.Module):
  """
  """
  def __init__(self,
    in_channels: int,
    out_channels: int) -> None:
    super(ResidualLayer, self).__init__()

    self.resblock = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
      nn.ReLU(),
      nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
    )

  """
  """
  def forward(self, input: torch.Tensor) -> torch.Tensor:
      return input + self.resblock(input)