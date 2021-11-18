import csv, json, os, tqdm
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from src.model.beta_vae import BetaVAE
from src.utils.early_stop import EarlyStop

## ------------------ HYPERPARAMETERS ------------------

DATA, EXP_NUMBER = "celeba", "5"
LOG_DIR = "experiment/beta-vae/" + DATA + EXP_NUMBER + "/"
DEVICE = torch.device("cuda:0")

BETA = 0.1

if DATA == "celeba":
  LATENT_DIM = 32

  ENC_MODEL_ARGS = [
    ("conv2d", {"in_channels": 3, "out_channels": 32, "kernel_size": 4, "stride": 2}),
    ("activation", "relu"),
    ("conv2d", {"in_channels": 32, "out_channels": 32, "kernel_size": 4, "stride": 2}),
    ("activation", "relu"),
    ("conv2d", {"in_channels": 32, "out_channels": 64, "kernel_size": 4, "stride": 2}),
    ("activation", "relu"),
    ("conv2d", {"in_channels": 64, "out_channels": 64, "kernel_size": 4, "stride": 2}),
    ("activation", "relu"),
    ("flatten", None),
    ("linear", {"in_features": 64*6*6, "out_features": 256}),
    ("activation", "relu"),
    ("linear", {"in_features": 256, "out_features": LATENT_DIM})
  ]

  DEC_LINEAR_MODEL_ARGS = [
    ("linear", {"in_features": LATENT_DIM, "out_features": 256}),
    ("activation", "relu"),
    ("linear", {"in_features": 256, "out_features": 64*8*8}),
    ("activation", "relu")
  ]

  DEC_SPATIAL_MODEL_ARGS = [
    ("upsample", {"scale_factor": 2}),
    ("conv2d", {"in_channels": 64, "out_channels": 64, "kernel_size": 3, "padding": 1}),
    ("activation", "relu"),
    ("upsample", {"scale_factor": 2}),
    ("conv2d", {"in_channels": 64, "out_channels": 32, "kernel_size": 3, "padding": 1}),
    ("activation", "relu"),
    ("upsample", {"scale_factor": 2}),
    ("conv2d", {"in_channels": 32, "out_channels": 32, "kernel_size": 3, "padding": 1}),
    ("activation", "relu"),
    ("upsample", {"scale_factor": 2}),
    ("conv2d", {"in_channels": 32, "out_channels": 3, "kernel_size": 3, "padding": 1}),
    ("activation", "sigmoid")
  ]

  DEC_LINEAR_TO_SPATIAL_SHAPE = (64, 8, 8)

elif DATA == "mnist":
  LATENT_DIM = 16

  ENC_MODEL_ARGS = [
    ("conv2d", {"in_channels": 1, "out_channels": 64, "kernel_size": 4, "stride": 2}),
    ("activation", "leakyrelu"),
    ("conv2d", {"in_channels": 64, "out_channels": 128, "kernel_size": 4, "stride": 2}),
    ("activation", "leakyrelu"),
    ("flatten", None),
    ("linear", {"in_features": 128*5*5, "out_features": 1024}),
    ("activation", "leakyrelu"),
    ("linear", {"in_features": 1024, "out_features": LATENT_DIM})
  ]

  DEC_LINEAR_MODEL_ARGS = [
    ("linear", {"in_features": LATENT_DIM, "out_features": 1024}),
    ("activation", "relu"),
    ("linear", {"in_features": 1024, "out_features": 128*7*7}),
  ]

  DEC_SPATIAL_MODEL_ARGS = [
    ("upsample", {"scale_factor": 2}),
    ("conv2d", {"in_channels": 128, "out_channels": 64, "kernel_size": 3, "padding": 1}),
    ("activation", "relu"),
    ("upsample", {"scale_factor": 2}),
    ("conv2d", {"in_channels": 64, "out_channels": 1, "kernel_size": 3, "padding": 1}),
    ("activation", "sigmoid")
  ]

  DEC_LINEAR_TO_SPATIAL_SHAPE = (128, 7, 7)

BATCH_SIZE = 128
EPOCHS = 100
LR = 1e-4
PATIENCE = 10

## instantiate model and optimizer
model = BetaVAE(BETA, ENC_MODEL_ARGS, DEC_LINEAR_MODEL_ARGS, DEC_SPATIAL_MODEL_ARGS, DEC_LINEAR_TO_SPATIAL_SHAPE, DEVICE)
model.to(DEVICE)
model_optim = optim.Adam(model.parameters(), LR)

## ------------------ DATALOADER ------------------

if DATA == "mnist":
  preprocessing = transforms.Compose([transforms.ToTensor()])

  ## take MNIST dataset using `torchvision.datasets` and split into train, valid, and test
  train_ds = datasets.MNIST("data", train=True, download=True, transform=preprocessing)
  valid_ds = datasets.MNIST("data", train=False, download=True, transform=preprocessing)

  ## get the size
  train_size = len(train_ds)
  valid_size = int(0.5 * len(valid_ds))
  test_size  = len(valid_ds) - valid_size

  ## split the dataset
  partition = [valid_size, test_size]
  valid_ds, test_ds = data.random_split(valid_ds, partition, torch.Generator().manual_seed(42))

elif DATA == "celeba":
  preprocessing = transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize(128),
    transforms.ToTensor()
  ])

  ## take the dataset under `data/celeba/img`
  ds = datasets.ImageFolder("data/celeba", transform=preprocessing)
  
  ## get the size
  train_size = int(0.8 * len(ds))
  valid_size = int(0.1 * len(ds))
  test_size = len(ds) - train_size - valid_size

  ## split the dataset
  partition = [train_size, valid_size, test_size]
  train_ds, valid_ds, test_ds = data.random_split(ds, partition, torch.Generator().manual_seed(42))

else:
  raise Exception("DATA is not recognized.")

## create the dataloader
train_dl = data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
valid_dl = data.DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=True)
test_dl = data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)

tqdm.tqdm.write("%s: %d training data, %d validation data, and %d test data." % (DATA, train_size, valid_size, test_size))

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
  "device": str(DEVICE),
  "patience": PATIENCE
}

os.makedirs(LOG_DIR)
with open(LOG_DIR + "config.json", "w") as f:
  json.dump(CONFIG, f)

## ------------------ TRAINING LOOP ------------------

train_loss_epochs = []
valid_loss_epochs = []

best_loss = float('Inf')
early_stopping = EarlyStop(patience=PATIENCE)

for epoch in tqdm.tqdm(range(EPOCHS)):
  train_loss = 0
  valid_loss = 0

  ## training loop
  model.train()
  for imgs, _ in tqdm.tqdm(train_dl, total=(train_size // BATCH_SIZE) + 1, leave=False, desc="Training"):
    model.zero_grad()
    imgs = imgs.to(DEVICE)
    
    loss = model(imgs)
    train_loss += loss.item() * imgs.shape[0]
    
    loss.backward()
    model_optim.step()
  
  ## validation loop
  model.eval()
  with torch.no_grad():
    for imgs, _ in tqdm.tqdm(valid_dl, total=(valid_size // BATCH_SIZE) + 1, leave=False, desc="Validation"):
      imgs = imgs.to(DEVICE)
      loss = model(imgs)
      valid_loss += loss.item() * imgs.shape[0]
  
  ## log the loss
  avg_train_loss = train_loss / train_size
  avg_valid_loss = valid_loss / valid_size

  train_loss_epochs.append(avg_train_loss)
  valid_loss_epochs.append(avg_valid_loss)

  tqdm.tqdm.write("Epoch %02d :: Tr-Loss %0.4f :: Vl-Loss %0.4f" %
    (epoch, train_loss_epochs[-1], valid_loss_epochs[-1]))

  ## log the model
  chkpt = {
    "model": model.state_dict(),
    "optim": model_optim.state_dict(),
    "epoch": epoch,
    "train_loss": train_loss,
    "valid_loss": valid_loss
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

tqdm.tqdm.write("Loading best model: epoch %02d & %0.4f validation loss" % (d["epoch"], d["valid_loss"]))

## testing loop using best model
test_loss = 0
with torch.no_grad():
  for imgs, _ in test_dl:
    imgs = imgs.to(DEVICE)
    loss = model(imgs)
    test_loss += loss.item() * imgs.shape[0]

tqdm.tqdm.write("Final Ts-Loss %0.4f" % (test_loss / test_size))

## log losses to csv
with open(LOG_DIR + "log.csv", "w") as f:
  writer = csv.writer(f)
  writer.writerow(['Training Loss', 'Validation Loss'])
  for tl, vl in zip(train_loss_epochs, valid_loss_epochs):
    writer.writerow(['%0.5f' % tl, '%0.5f' % vl])
  writer.writerow(['%0.5f' % (test_loss / test_size)])

## ------------------ RECONSTRUCTION TEST ------------------

if DATA == "mnist":
  ## this is carefully picked to cover 0-9 class
  idx = [1, 12, 24, 44, 47, 48, 52, 78, 84, 116]
  cmap = 'gray'

elif DATA == "celeba":
  ## carefully picked to compare with other hyperparams
  idx = [8, 16, 32, 64]
  cmap = None

## bacth the original image
original_img = torch.cat([test_ds[i][0].unsqueeze(0) for i in idx]).to(DEVICE)

## reconstruction
var_posterior = model.encoder(original_img)
reconstructed_img = model.decoder(var_posterior.mean)

## change shape and convert to numpy to be visualized
if DATA == "mnist":
  original_img = original_img.squeeze(1)
  reconstructed_img = reconstructed_img.squeeze(1)

elif DATA == "celeba":
  original_img = original_img.moveaxis(1, -1)
  reconstructed_img = reconstructed_img.moveaxis(1, -1)

original_img = original_img.detach().cpu().numpy()
reconstructed_img = reconstructed_img.detach().cpu().numpy()

## plot
n_row, n_col = 2, len(idx)
f, axs = plt.subplots(n_row, n_col, figsize=(n_col*3, n_row*4))
axs[0, n_col//2].set_title("Original Image")
axs[1, n_col//2].set_title("Reconstructed Image")
for j in range(len(idx)):
  axs[0, j].imshow(original_img[j], cmap=cmap)
  axs[1, j].imshow(reconstructed_img[j], cmap=cmap)
  axs[0, j].axis('off')
  axs[1, j].axis('off')
  
f.savefig(LOG_DIR + "reconstruction.png")
plt.close(f)