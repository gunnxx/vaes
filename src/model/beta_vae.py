from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.distributions as d

from src.model.decoder import Decoder
from src.model.gaussian_encoder import GaussianEncoder
from src.utils.common import model_args_dtype

class BetaVAE(nn.Module):
  """
  """
  def __init__(self,
    beta: float,
    enc_model_args: model_args_dtype,
    dec_linear_model_args: model_args_dtype,
    dec_spatial_model_args: model_args_dtype,
    dec_linear_to_spatial_shape: Tuple[int, int, int]) -> None:
    super(BetaVAE, self).__init__()

    ## to weight losses
    self.beta = beta
    self.loss_fn = nn.BCELoss()
      
    ## encoder and decoder
    self.encoder = GaussianEncoder(enc_model_args)
    self.decoder = Decoder(dec_linear_model_args,
      dec_linear_to_spatial_shape, dec_spatial_model_args)
    
    ## normal prior as variational posterior target
    latent_dim = dec_linear_model_args[0][1]["in_features"]
    mu = torch.zeros(latent_dim)
    std = torch.ones(latent_dim)
    self.prior = d.Independent(d.Normal(mu, std), 1)
  
  """
  """
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    ## encoding
    posterior = self.encoder.forward(x)

    ## sampling
    if self.training:
      sample = posterior.rsample()
    else:
      sample = posterior.mean
    
    ## decoding
    reconstruction = self.decoder.forward(sample)

    ## compute loss
    kld_loss = d.kl.kl_divergence(posterior, self.prior)
    nll_loss = self.loss_fn(reconstruction, x)
    loss = self.beta * kld_loss + nll_loss

    return loss.mean()