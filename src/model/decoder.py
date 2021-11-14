from typing import Tuple

import torch
import torch.distributions as d
import torch.nn as nn

from src.utils.common import model_args_dtype, instantiate_layer

class Decoder(nn.Module):  
  """
  """
  def __init__(self,
    linear_model_args: model_args_dtype,
    linear_to_spatial_shape: Tuple[int, int, int],
    spatial_model_args: model_args_dtype) -> None:
    super(Decoder, self).__init__()
    
    self.linear_layers = nn.Sequential([instantiate_layer(lt, lp) for lt, lp in linear_model_args])
    self.spatial_layers = nn.Sequential([instantiate_layer(lt, lp) for lt, lp in spatial_model_args])
    self.reshape_size = linear_to_spatial_shape
  
  """
  """
  def forward(self, x: torch.Tensor) -> d.Bernoulli:
    h = self.linear_layers(x)
    h = h.view(-1, *self.reshape_size)
    h = self.spatial_layers(h)
    return h