import csv, itertools, json, os, tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from src.model.vector_quantizer import VectorQuantizer
from src.utils.common import instantiate_layer

## ------------------ HYPERPARAMETERS ------------------

DATA, EXP_NUMBER = "celeba", "0"
LOG_DIR = "experiment/vq-vae/" + DATA + EXP_NUMBER + "/"
DEVICE = torch.device("cpu")

BETA = 0.25

if DATA == "mnist":
  NUM_EMBEDDING = 128
  EMBEDDING_DIM = 3
  NUM_IN_CHANNEL = 1

elif DATA == "celeba":
  NUM_EMBEDDING = 512
  EMBEDDING_DIM = 1
  NUM_IN_CHANNEL = 3

elif DATA == "cifar10":
  NUM_EMBEDDING = 256
  EMBEDDING_DIM = 10
  NUM_IN_CHANNEL = 3

ENC_MODEL_ARGS = [
  ("conv2d", {"in_channels": NUM_IN_CHANNEL, "out_channels": 32, "kernel_size": 4, "stride": 2, "padding": 1}),
  ("activation", "relu"),
  ("conv2d", {"in_channels": 32, "out_channels": EMBEDDING_DIM, "kernel_size": 4, "stride": 2, "padding": 1}),
  ("activation", "relu"),
  ("residuallayer", {"in_channels": EMBEDDING_DIM, "out_channels": EMBEDDING_DIM}),
  ("activation", "relu"),
  ("residuallayer", {"in_channels": EMBEDDING_DIM, "out_channels": EMBEDDING_DIM}),
  ("activation", "relu")
]

DEC_MODEL_ARGS = [
  ("residuallayer", {"in_channels": EMBEDDING_DIM, "out_channels": EMBEDDING_DIM}),
  ("activation", "relu"),
  ("residuallayer", {"in_channels": EMBEDDING_DIM, "out_channels": EMBEDDING_DIM}),
  ("activation", "relu"),
  ("convtranspose2d", {"in_channels": EMBEDDING_DIM, "out_channels": 32, "kernel_size": 4, "stride": 2, "padding": 1}),
  ("activation", "relu"),
  ("convtranspose2d", {"in_channels": 32, "out_channels": NUM_IN_CHANNEL, "kernel_size": 4, "stride": 2, "padding": 1}),
  ("activation", "sigmoid")
]

BATCH_SIZE = 128
EPOCHS = 1
LR = 2e-4

## instantiate model and optimizer
encoder = nn.Sequential(*[instantiate_layer(ltype, lparams) for ltype, lparams in ENC_MODEL_ARGS])
decoder = nn.Sequential(*[instantiate_layer(ltype, lparams) for ltype, lparams in DEC_MODEL_ARGS])
vqlayer = VectorQuantizer(NUM_EMBEDDING, EMBEDDING_DIM, BETA)

encoder.to(DEVICE)
decoder.to(DEVICE)
vqlayer.to(DEVICE)

model_optim = optim.Adam(itertools.chain(encoder.parameters(), decoder.parameters(), vqlayer.parameters()), LR)

## ------------------ DATALOADER ------------------

if DATA == "mnist":
  preprocessing = transforms.Compose([transforms.ToTensor()])

  ## take MNIST dataset using `torchvision.datasets`
  train_ds = datasets.MNIST("data", train=True, download=True, transform=preprocessing)
  valid_ds = datasets.MNIST("data", train=False, download=True, transform=preprocessing)

  ## get the size
  train_size = len(train_ds)
  valid_size  = len(valid_ds)

elif DATA == "celeba":
  preprocessing = transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize(128),
    transforms.ToTensor()
  ])

  ## take the dataset under `data/celeba/img`
  ds = datasets.ImageFolder("data/celeba", transform=preprocessing)
  
  ## get the size
  train_size = int(0.85 * len(ds))
  valid_size = len(ds) - train_size

  ## split the dataset
  partition = [train_size, valid_size]
  train_ds, valid_ds = data.random_split(ds, partition, torch.Generator().manual_seed(42))

elif DATA == "cifar10":
  preprocessing = transforms.Compose([transforms.ToTensor()])

  ## take MNIST dataset using `torchvision.datasets`
  train_ds = datasets.CIFAR10("data", train=True, download=True, transform=preprocessing)
  valid_ds = datasets.CIFAR10("data", train=False, download=True, transform=preprocessing)

  ## get the size
  train_size = len(train_ds)
  valid_size  = len(valid_ds)

else:
  raise Exception("DATA is not recognized.")

## create the dataloader
train_dl = data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
valid_dl = data.DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=True)

## ------------------ LOG CONFIGURATION ------------------

CONFIG = {
  "beta": BETA,
  "num_embedding": NUM_EMBEDDING,
  "embedding_dim": EMBEDDING_DIM,
  "enc_model_args": ENC_MODEL_ARGS,
  "dec_model_args": DEC_MODEL_ARGS,
  "batch_size": BATCH_SIZE,
  "epochs": EPOCHS,
  "learning_rate": LR,
  "device": str(DEVICE)
}

os.makedirs(LOG_DIR)
with open(LOG_DIR + "config.json", "w") as f:
  json.dump(CONFIG, f)

## ------------------ TRAINING LOOP ------------------

train_total_loss_epochs = []
train_recon_loss_epochs = []
train_embed_loss_epochs = []

best_loss = float('Inf')

## training loop
encoder.train()
decoder.train()
vqlayer.train()

for epoch in tqdm.tqdm(range(EPOCHS)):
  train_total_loss = 0
  train_recon_loss = 0
  train_embed_loss = 0
  
  for imgs, _ in tqdm.tqdm(train_dl, total=int(train_size // BATCH_SIZE) + 1, leave=False):
    model_optim.zero_grad()
    imgs = imgs.to(DEVICE)
    
    ## TODO: straight through estimator
    ze = encoder(imgs)
    zq, vq_loss = vqlayer(ze)
    recon_imgs = decoder(zq)
    recon_loss = F.binary_cross_entropy(recon_imgs, imgs)

    ## optimize the total loss
    loss = recon_loss + vq_loss
    loss.backward()
    model_optim.step()

    train_total_loss += loss.item() * imgs.shape[0]
    train_recon_loss += recon_loss.item() * imgs.shape[0]
    train_embed_loss += vq_loss.item() * imgs.shape[0]
  
  ## log the loss
  avg_train_loss = train_total_loss / train_size
  train_total_loss_epochs.append(avg_train_loss)
  train_recon_loss_epochs.append(train_recon_loss / train_size)
  train_embed_loss_epochs.append(train_embed_loss / train_size)

  tqdm.tqdm.write("Epoch %02d :: Tr-Loss %0.4f" % (epoch, avg_train_loss))

  ## log the model
  chkpt = {
    "encoder": encoder.state_dict(),
    "decoder": decoder.state_dict(),
    "vqlayer": vqlayer.state_dict(),
    "optim": model_optim.state_dict(),
    "epoch": epoch,
    "loss": avg_train_loss
  }

  ## latest model
  torch.save(chkpt, LOG_DIR + "latest.pt")

  ## best model
  if avg_train_loss < best_loss:
    best_loss = avg_train_loss
    torch.save(chkpt, LOG_DIR + "best.pt")

## load best model
d = torch.load(LOG_DIR + "best.pt", map_location=DEVICE)

encoder.load_state_dict(d["encoder"])
decoder.load_state_dict(d["decoder"])
vqlayer.load_state_dict(d["vqlayer"])

encoder.eval()
decoder.eval()
vqlayer.eval()

tqdm.tqdm.write("Loading best model: epoch %02d & %0.4f loss" % (d["epoch"], d["loss"]))

## testing loop using best model
valid_total_loss = 0
valid_recon_loss = 0
valid_embed_loss = 0

with torch.no_grad():
  for imgs, _ in valid_dl:
    imgs = imgs.to(DEVICE)

    ## TODO: straight through estimator
    ze = encoder(imgs)
    zq, vq_loss = vqlayer(ze)
    recon_imgs = decoder(zq)
    recon_loss = F.binary_cross_entropy(recon_imgs, imgs)

    valid_total_loss += loss.item() * imgs.shape[0]
    valid_recon_loss += recon_loss.item() * imgs.shape[0]
    valid_embed_loss += vq_loss.item() * imgs.shape[0]

tqdm.tqdm.write("Final Ts-Loss %0.4f" % (valid_total_loss / valid_size))

## log losses to csv
with open(LOG_DIR + "log.csv", "w") as f:
  writer = csv.writer(f)
  writer.writerow(['TotalLoss', 'ReconLoss', 'EmbedLoss'])
  
  for tl, rl, el in zip(train_total_loss_epochs, train_recon_loss_epochs, train_embed_loss_epochs):
    writer.writerow(('%0.5f %0.5f %0.5f' % (tl, rl, el)).split())
  
  writer.writerow(('%0.5f %0.5f %0.5f' % (
    valid_total_loss / valid_size, valid_recon_loss / valid_size, valid_embed_loss / valid_size
  )).split())

## ------------------ RECONSTRUCTION TEST ------------------

if DATA == "mnist":
  ## this is carefully picked to cover 0-9 class
  idx = [1, 12, 24, 44, 47, 48, 52, 78, 84, 116]
  cmap = 'gray'

elif DATA == "celeba" or DATA == "cifar10":
  ## carefully picked to compare with other hyperparams
  idx = [0, 2, 4, 8, 16, 32]
  cmap = None

with torch.no_grad():
  ## bacth the original image
  imgs = torch.cat([valid_ds[i][0].unsqueeze(0) for i in idx])

  ## reconstruction
  ze = encoder(imgs)
  zq, _ = vqlayer(ze)
  recon_imgs = decoder(zq)

## change shape and convert to numpy to be visualized
if DATA == "mnist":
  imgs = imgs.squeeze(1)
  recon_imgs = recon_imgs.squeeze(1)

elif DATA == "celeba" or DATA == "cifar10":
  imgs = imgs.moveaxis(1, -1)
  recon_imgs = recon_imgs.moveaxis(1, -1)

imgs = imgs.detach().cpu().numpy()
recon_imgs = recon_imgs.detach().cpu().numpy()

## plot
n_row, n_col = 2, len(idx)
f, axs = plt.subplots(n_row, n_col, figsize=(n_col*3, n_row*4))
axs[0, n_col//2].set_title("Original Image")
axs[1, n_col//2].set_title("Reconstructed Image")
for j in range(len(idx)):
  axs[0, j].imshow(imgs[j], cmap=cmap)
  axs[1, j].imshow(recon_imgs[j], cmap=cmap)
  axs[0, j].axis('off')
  axs[1, j].axis('off')
  
f.savefig(LOG_DIR + "reconstruction.png")
plt.close(f)