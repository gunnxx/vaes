import torch
import torch.nn as nn

from src.utils.common import model_args_dtype, instantiate_layer

class InfoDiscriminator(nn.Module):
  """
  """
  def __init__(self,
    base_model_args: model_args_dtype,
    data: str,
    device: torch.device) -> None:
    super(InfoDiscriminator, self).__init__()

    ## ordinary discriminator
    base_out_size = base_model_args[-3][1]["out_features"]
    self.base = nn.Sequential(*[instantiate_layer(lt, lp) for lt, lp in base_model_args])
    self.head = nn.Sequential(nn.Linear(base_out_size, 1), nn.Sigmoid())

    ## based on InfoGAN paper
    if data == "mnist":
      self.code_heads = [
        nn.Sequential(nn.Linear(base_out_size, 10), nn.Softmax(dim=-1)).to(device),
        nn.Sequential(nn.Linear(base_out_size, 2)).to(device)
      ]

    elif data == "celeba":
      self.code_heads = [
        nn.Sequential(nn.Linear(base_out_size, 10), nn.Softmax(dim=-1)).to(device) for _ in range(10)
      ]
  
  """
  """
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    h = self.base(x)
    return self.head(h), [code_head(h) for code_head in self.code_heads]