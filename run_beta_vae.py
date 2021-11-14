import json, os, tqdm
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from src.model.beta_vae import BetaVAE
from src.utils.early_stop import EarlyStop

## ------------------ HYPERPARAMETERS ------------------

BETA = 1.
LATENT_DIM = 16

ENC_MODEL_ARGS = [
  ("conv2d", {"in_channels": 1, "out_channels": 32, "kernel_size": 4, "stride": 2}),
  ("activation", "relu"),
  ("conv2d", {"in_channels": 32, "out_channels": 64, "kernel_size": 4, "stride": 2}),
  ("activation", "relu"),
  ("conv2d", {"in_channels": 64, "out_channels": 64, "kernel_size": 4, "stride": 2}),
  ("activation", "relu"),
  ("flatten", None),
  ("linear", {"in_features": 64, "out_features": LATENT_DIM})
]

DEC_LINEAR_MODEL_ARGS = [
  ("linear", {"in_features": LATENT_DIM, "out_features": 64}),
  ("activation", "relu")
]

DEC_SPATIAL_MODEL_ARGS = [
  ("upsample", {"scale_factor": 4}),
  ("conv2d", {"in_channels": 64, "out_channels": 64, "kernel_size": 3, "padding": "same"}),
  ("activation", "relu"),
  ("upsample", {"scale_factor": 4}),
  ("conv2d", {"in_channels": 64, "out_channels": 32, "kernel_size": 3, "padding": "same"}),
  ("activation", "relu"),
  ("upsample", {"scale_factor": 2}),
  ("conv2d", {"in_channels": 32, "out_channels": 32, "kernel_size": 3, "padding": "valid"}),
  ("activation", "relu"),
  ("conv2d", {"in_channels": 32, "out_channels": 1, "kernel_size": 3, "padding": "valid"}),
  ("activation", "sigmoid")
]

DEC_LINEAR_TO_SPATIAL_SHAPE = (64, 1, 1)

DATA, EXP_NUMBER = "mnist", "0"
LOG_DIR = "experiment/beta-vae/" + DATA + EXP_NUMBER + "/"

DEVICE = torch.device("cpu")
BATCH_SIZE = 128
EPOCHS = 2
LR = 1e-4

## instantiate model and optimizer
model = BetaVAE(BETA, ENC_MODEL_ARGS, DEC_LINEAR_MODEL_ARGS,
  DEC_SPATIAL_MODEL_ARGS, DEC_LINEAR_TO_SPATIAL_SHAPE).to(DEVICE)
model_optim = optim.Adam(model.parameters(), LR)

## ------------------ DATALOADER ------------------

preprocessing = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize(0, 255)
])

if DATA == "mnist":
  ## take MNIST dataset using `torchvision.datasets` and split into train, valid, and test
  train_ds = datasets.MNIST("data", train=True, download=True, transform=preprocessing)
  valid_ds = datasets.MNIST("data", train=False, download=True, transform=preprocessing)
  valid_len = int(0.5 * len(valid_ds))
  test_len  = len(valid_ds) - valid_len
  valid_ds, test_ds = data.random_split(valid_ds, [valid_len, test_len])

  ## take only the image part
  train_ds = torch.cat([img.unsqueeze(0) for img, _ in train_ds]).to(DEVICE)
  valid_ds = torch.cat([img.unsqueeze(0) for img, _ in valid_ds]).to(DEVICE)
  test_ds = torch.cat([img.unsqueeze(0) for img, _ in test_ds]).to(DEVICE)

## create the dataloader
train_dl = data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
valid_dl = data.DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=True)
test_dl = data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)

## ------------------ LOG CONFIGURATION ------------------

CONFIG = {
  "beta": BETA,
  "latent_dim": LATENT_DIM,
  "enc_model_args": ENC_MODEL_ARGS,
  "dec_linear_model_args": DEC_LINEAR_MODEL_ARGS,
  "dec_spatial_model_args": DEC_SPATIAL_MODEL_ARGS,
  "dec_linear_to_spatial_shape": DEC_LINEAR_TO_SPATIAL_SHAPE,
  "batch_size": BATCH_SIZE,
  "epochs": EPOCHS,
  "learning_rate": LR,
  "log_dir": LOG_DIR,
  "device": DEVICE.type
}

os.makedirs(LOG_DIR)
with open(LOG_DIR + "config.json", "w") as f:
  json.dump(CONFIG, f)

## ------------------ TRAINING LOOP ------------------

train_loss_epochs = []
valid_loss_epochs = []

best_loss = np.inf
early_stopping = EarlyStop(patience=3, warmup=10)

for epoch in tqdm.tqdm(range(EPOCHS)):
  train_losses = []
  valid_losses = []

  ## training loop
  model.train()
  for batch in train_dl:
    model.zero_grad()
    
    train_loss = model(batch)
    train_losses.append(train_loss.item())
    
    train_loss.backward()
    model_optim.step()
  
  ## validation loop
  model.eval()
  with torch.no_grad():
    for batch in valid_dl:
      valid_loss = model(batch)
      valid_losses.append(valid_loss.item())
  
  ## log the loss
  avg_train_loss = np.array(train_losses).mean()
  avg_valid_loss = np.array(valid_losses).mean()

  train_loss_epochs.append(avg_train_loss.item())
  valid_loss_epochs.append(avg_valid_loss.item())

  tqdm.tqdm.write("Epoch %02d :: Tr-Loss %0.4f :: Vl-Loss %0.4f" %
    (epoch, train_loss_epochs[-1], valid_loss_epochs[-1]))

  ## log the model
  chkpt = {
    "model": model.state_dict(),
    "optim": model_optim.state_dict(),
    "epoch": epoch,
    "loss": train_loss
  }

  ## latest model
  torch.save(chkpt, LOG_DIR + "latest.pt")

  ## best model
  if avg_valid_loss < best_loss:
    best_loss = avg_valid_loss
    torch.save(chkpt, LOG_DIR + "best.pt")
  
  ## check for early stopping
  early_stopping(avg_valid_loss)
  if early_stopping.is_stop():
    tqdm.tqdm.write("Early stop is called!")
    break

## load best model
d = torch.load(LOG_DIR + "best.pt", map_location=DEVICE)
model.load_state_dict(d["model"])
model.eval()

tqdm.tqdm.write("Loading best model: epoch %02d & %0.4f validation loss" %
  (d["epoch"], d["loss"].item()))

## testing loop using best model
test_losses = []
with torch.no_grad():
  for batch in test_dl:
    test_loss = model(batch)
    test_losses.append(test_loss.item())

tqdm.tqdm.write("Final Ts-Loss %0.4f" % np.array(test_losses).mean().item())

## ------------------ RECONSTRUCTION TEST ------------------

if DATA == "mnist":
  ## this is carefully picked to cover 0-9 class
  idx = [0, 8, 20, 22, 121, 124, 125, 130, 133, 137]

  ## bacth the original image
  original_img = torch.cat([test_ds[i].unsqueeze(0) for i in idx])

  ## reconstruction
  var_posterior = model.encoder(original_img)
  reconstructed_img = model.decoder(var_posterior.mean)

  ## change to numpy to be visualized
  original_img = original_img.detach().cpu().numpy()
  reconstructed_img = reconstructed_img.detach().cpu().numpy()

  ## plot
  n_row, n_col = 2, 10
  f, axs = plt.subplots(n_row, n_col, figsize=(16, 6))
  axs[0, n_col//2].set_title("Original Image")
  axs[1, n_col//2].set_title("Reconstructed Image")
  for j in range(len(idx)):
    axs[0, j].imshow(original_img[j, 0], cmap='gray')
    axs[1, j].imshow(reconstructed_img[j, 0], cmap='gray')
    axs[0, j].axis('off')
    axs[1, j].axis('off')
  
  f.savefig(LOG_DIR + "reconstruction.png")
  plt.close(f)