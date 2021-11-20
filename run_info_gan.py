import csv, itertools, json, os, tqdm
from torchvision.datasets.mnist import MNIST
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from src.model.decoder import Decoder
from src.model.info_discriminator import InfoDiscriminator

## ------------------ HYPERPARAMETERS ------------------

DATA, EXP_NUMBER = "mnist", "0"
LOG_DIR = "experiment/info-gan/" + DATA + EXP_NUMBER + "/"
DEVICE = torch.device("cpu")

NOISE_DIM = 16

if DATA == "mnist":
  CODE_DIM = 12

  DISC_MODEL_ARGS = [
    ("conv2d", {"in_channels": 1, "out_channels": 64, "kernel_size": 4, "stride": 2, "padding": 1}),
    ("activation", "leakyrelu"),
    ("conv2d", {"in_channels": 64, "out_channels": 128, "kernel_size": 4, "stride": 2, "padding": 1}),
    ("batchnorm2d", {"num_features": 128}),
    ("activation", "leakyrelu"),
    ("flatten", None),
    ("linear", {"in_features": 128*7*7, "out_features": 1024}),
    ("batchnorm1d", {"num_features": 1024}),
    ("activation", "leakyrelu")
  ]

  GEN_LINEAR_MODEL_ARGS = [
    ("linear", {"in_features": NOISE_DIM + CODE_DIM, "out_features": 1024}),
    ("batchnorm1d", {"num_features": 1024}),
    ("activation", "relu"),
    ("linear", {"in_features": 1024, "out_features": 128*7*7}),
    ("batchnorm1d", {"num_features": 128*7*7}),
    ("activation", "relu")
  ]

  GEN_SPATIAL_MODEL_ARGS = [
    ("convtranspose2d", {"in_channels": 128, "out_channels": 64, "kernel_size": 4, "stride": 2, "padding": 1}),
    ("batchnorm2d", {"num_features": 64}),
    ("activation", "relu"),
    ("convtranspose2d", {"in_channels": 64, "out_channels": 1, "kernel_size": 4, "stride": 2, "padding": 1}),
    ("activation", "tanh")
  ]

  GEN_LINEAR_TO_SPATIAL_SHAPE = (128, 7, 7)

elif DATA == "celeba":
  CODE_DIM = 100

  DISC_MODEL_ARGS = [
    ("conv2d", {"in_channels": 3, "out_channels": 32, "kernel_size": 4, "stride": 2, "padding": 1}),
    ("activation", "leakyrelu"),
    ("conv2d", {"in_channels": 32, "out_channels": 32, "kernel_size": 4, "stride": 2, "padding": 1}),
    ("batchnorm2d", {"num_features": 32}),
    ("activation", "leakyrelu"),
    ("conv2d", {"in_channels": 32, "out_channels": 64, "kernel_size": 4, "stride": 2, "padding": 1}),
    ("batchnorm2d", {"num_features": 64}),
    ("activation", "leakyrelu"),
    ("conv2d", {"in_channels": 64, "out_channels": 64, "kernel_size": 4, "stride": 2, "padding": 1}),
    ("batchnorm2d", {"num_features": 64}),
    ("activation", "leakyrelu"),
    ("flatten", None),
    ("linear", {"in_features": 64*8*8, "out_features": 1024}),
    ("batchnorm1d", {"num_features": 1024}),
    ("activation", "leakyrelu"),
  ]

  GEN_LINEAR_MODEL_ARGS = [
    ("linear", {"in_features": NOISE_DIM + CODE_DIM, "out_features": 1024}),
    ("batchnorm1d", {"num_features": 1024}),
    ("activation", "relu"),
    ("linear", {"in_features": 1024, "out_features": 64*8*8}),
    ("batchnorm1d", {"num_features": 64*8*8}),
    ("activation", "relu")
  ]

  GEN_SPATIAL_MODEL_ARGS = [
    ("convtranspose2d", {"in_channels": 64, "out_channels": 64, "kernel_size": 4, "stride": 2, "padding": 1}),
    ("batchnorm2d", {"num_features": 64}),
    ("activation", "relu"),
    ("convtranspose2d", {"in_channels": 64, "out_channels": 32, "kernel_size": 4, "stride": 2, "padding": 1}),
    ("batchnorm2d", {"num_features": 32}),
    ("activation", "relu"),
    ("convtranspose2d", {"in_channels": 32, "out_channels": 32, "kernel_size": 4, "stride": 2, "padding": 1}),
    ("batchnorm2d", {"num_features": 32}),
    ("activation", "relu"),
    ("convtranspose2d", {"in_channels": 32, "out_channels": 3, "kernel_size": 4, "stride": 2, "padding": 1}),
    ("activation", "tanh")
  ]

  GEN_LINEAR_TO_SPATIAL_SHAPE = (64, 8, 8)

BATCH_SIZE = 128
EPOCHS = 1
G_LR, D_LR, I_LR = 1e-4, 1e-4, 1e-4

## ------------------ INSTANTIATE MODEL AND OPTIM ------------------

def weights_init_normal(m):
  classname = m.__class__.__name__
  if classname.find("Conv") != -1:
    torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
  elif classname.find("BatchNorm") != -1:
    torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    torch.nn.init.constant_(m.bias.data, 0.0)

generator = Decoder(GEN_LINEAR_MODEL_ARGS, GEN_LINEAR_TO_SPATIAL_SHAPE, GEN_SPATIAL_MODEL_ARGS)
discriminator = InfoDiscriminator(DISC_MODEL_ARGS, DATA, DEVICE)

generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

generator.to(DEVICE)
discriminator.to(DEVICE)

generator_optim = optim.Adam(generator.parameters(), G_LR)
discriminator_optim = optim.Adam(discriminator.parameters(), D_LR)
information_optim = optim.Adam(
  itertools.chain(generator.parameters(), discriminator.parameters()), I_LR)

## ------------------ DATALOADER ------------------

if DATA == "mnist":
  preprocessing = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
  ])

  ## take MNIST dataset using `torchvision.datasets` and split into train, valid, and test
  train_ds = datasets.MNIST("data", train=True, download=True, transform=preprocessing)
  test_ds = datasets.MNIST("data", train=False, download=True, transform=preprocessing)

  ## get the size
  train_size = len(train_ds)
  test_size = len(test_ds)

elif DATA == "celeba":
  preprocessing = transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
  ])

  ## take the dataset under `data/celeba/img`
  ds = datasets.ImageFolder("data/celeba", transform=preprocessing)
  
  ## get the size
  train_size = int(0.85 * len(ds))
  test_size = len(ds) - train_size

  ## split the dataset
  partition = [train_size, test_size]
  train_ds, test_ds = data.random_split(ds, partition, torch.Generator().manual_seed(42))

else:
  raise Exception("DATA is not recognized.")

## create the dataloader
train_dl = data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_dl = data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)

tqdm.tqdm.write("Data loaded: %d training data and %d testing data." % (train_size, test_size))

## ------------------ LOG CONFIGURATION ------------------

CONFIG = {
  "noise_dim": NOISE_DIM,
  "disc_model_args": DISC_MODEL_ARGS,
  "gen_linear_model_args": GEN_LINEAR_MODEL_ARGS,
  "gen_spatial_model_args": GEN_SPATIAL_MODEL_ARGS,
  "gen_linear_to_spatial_shape": GEN_LINEAR_TO_SPATIAL_SHAPE,
  "batch_size": BATCH_SIZE,
  "epochs": EPOCHS,
  "generator_learning_rate": G_LR,
  "discriminator_learning_rate": D_LR,
  "information_learning_rate": I_LR,
  "device": str(DEVICE)
}

os.makedirs(LOG_DIR)
with open(LOG_DIR + "config.json", "w") as f:
  json.dump(CONFIG, f)

## -------------- HELPER FUNCTION TO GENERATE CODES --------------

def to_flatten_categorical(x, num_category, device):
  batch_size, code_size = x.shape
  
  ## idx to flip the values to 1.
  batch_idx = np.repeat(list(range(batch_size)), code_size)
  code_idx = list(range(code_size)) * batch_size
  data_idx = x.flatten()
  
  ## categorical data
  cat = torch.zeros(batch_size, code_size, num_category, device=device)
  cat[batch_idx, code_idx, data_idx] = 1.
  
  return cat.reshape(-1, code_size * num_category)

def generate_noise_and_codes(noise_dim, batch_size, data, device):
  z = torch.randn(batch_size, noise_dim, device=device)

  if data == "mnist":
    c1 = torch.randint(0, 10, (batch_size, 1), device=device)
    c1 = to_flatten_categorical(c1, 10, device=device)
    c2 = torch.rand(batch_size, 2, device=device) * 2 - 1
    c = [c1, c2]
  
  elif data == "celeba":
    c = torch.randint(0, 10, (batch_size, 10), device=device)
    c = to_flatten_categorical(c, 10, device=device)
    c = c.split(10, -1)
  
  return [z, *c]

## ------------------ TRAINING LOOP ------------------

g_loss_epochs = []
d_loss_epochs = []
i_loss_epochs = []
t_loss_epochs = []

best_g_loss = float('inf')
best_d_loss = float('inf')
best_i_loss = float('inf')
best_t_loss = float('inf')

## losses
discriminating_loss = nn.BCELoss()
if DATA == "mnist":
  code_loss = [nn.CrossEntropyLoss(), nn.MSELoss()]
elif DATA == "celeba":
  code_loss = [nn.CrossEntropyLoss() for _ in range(10)]

for epoch in tqdm.tqdm(range(EPOCHS)):
  g_loss_ = 0
  d_loss_ = 0
  i_loss_ = 0

  ## training loop
  generator.train()
  discriminator.train()

  for imgs, _ in tqdm.tqdm(train_dl, total=(train_size // BATCH_SIZE) + 1, leave=False):
    bsz = imgs.shape[0]
    imgs = imgs.to(DEVICE)
    real = torch.ones(bsz, 1, device=DEVICE)
    fake = torch.zeros(bsz, 1, device=DEVICE)

    ## ---- TRAINING GENERATOR ----

    generator_optim.zero_grad()

    ## generate images
    gen_in = generate_noise_and_codes(NOISE_DIM, bsz, DATA, DEVICE)
    generated_imgs = generator(torch.cat(gen_in, dim=-1))

    ## we want the generated images to fool the discriminator
    g_pred, _ = discriminator(generated_imgs)
    g_loss = discriminating_loss(g_pred, real)
    
    g_loss.backward()
    generator_optim.step()

    ## ---- TRAINING DISCRIMINATOR ----

    discriminator_optim.zero_grad()

    ## loss for real images
    r_pred, _ = discriminator(imgs)
    r_loss = discriminating_loss(r_pred, real)

    ## loss for fake images
    g_pred, _ = discriminator(generated_imgs.detach())
    f_loss = discriminating_loss(g_pred, fake)

    ## total discriminator loss
    d_loss = (r_loss + f_loss) / 2

    d_loss.backward()
    discriminator_optim.step()

    ## ---- TRAINING INFORMATION ---- 

    information_optim.zero_grad()

    ## predict the code
    gen_in = generate_noise_and_codes(NOISE_DIM, bsz, DATA, DEVICE)
    generated_imgs = generator(torch.cat(gen_in, dim=-1))
    _, code_pred = discriminator(generated_imgs)

    ## compute info loss
    info_loss = 0
    for ct, cp, l in zip(gen_in[1:], code_pred, code_loss):
      if isinstance(l, nn.CrossEntropyLoss): _, ct = ct.max(1)
      info_loss += l(cp, ct)
    
    info_loss.backward()
    information_optim.step()

    ## accumulate losses for logging
    g_loss_ += g_loss.item() * bsz
    d_loss_ += d_loss.item() * bsz
    i_loss_ += info_loss.item() * bsz

    tqdm.tqdm.write("G %0.4f :: D %0.4f :: I %0.4f" % (g_loss.item(), d_loss.item(), info_loss.item()))
  
  ## log the loss
  avg_g_loss = g_loss_ / train_size
  avg_d_loss = d_loss_ / train_size
  avg_i_loss = i_loss_ / train_size
  avg_t_loss = avg_g_loss + avg_d_loss + avg_i_loss

  g_loss_epochs.append(avg_g_loss)
  d_loss_epochs.append(avg_d_loss)
  i_loss_epochs.append(avg_i_loss)
  t_loss_epochs.append(avg_t_loss)

  tqdm.tqdm.write("Epoch %02d :: GLoss %0.4f :: DLoss %0.4f :: ILoss %0.4f :: TotalAvg %0.4f" %
    (epoch, avg_g_loss, avg_d_loss, avg_i_loss, avg_t_loss))

  ## log the model
  chkpt = {
    "generator": generator.state_dict(),
    "discriminator": discriminator.state_dict(),
    "generator_optim": generator_optim.state_dict(),
    "discriminator_optim": discriminator_optim.state_dict(),
    "information_optim": information_optim.state_dict(),
    "generator_loss": avg_g_loss,
    "discriminator_loss": avg_d_loss,
    "information_loss": avg_i_loss,
    "total_loss": avg_t_loss,
    "epoch": epoch
  }

  ## latest model
  torch.save(chkpt, LOG_DIR + "latest.pt")

  ## best model
  if avg_g_loss < best_g_loss:
    best_g_loss = avg_g_loss
    torch.save(chkpt, LOG_DIR + "best_g.pt")
  
  if avg_d_loss < best_d_loss:
    best_d_loss = avg_d_loss
    torch.save(chkpt, LOG_DIR + "best_d.pt")
  
  if avg_i_loss < best_i_loss:
    best_i_loss = avg_i_loss
    torch.save(chkpt, LOG_DIR + "best_i.pt")
  
  if avg_t_loss < best_t_loss:
    best_t_loss = avg_t_loss
    torch.save(chkpt, LOG_DIR + "best_t.pt")

## load best model
d = torch.load(LOG_DIR + "best_t.pt", map_location=DEVICE)

generator.load_state_dict(d["generator"])
discriminator.load_state_dict(d["discriminator"])

generator.eval()
discriminator.eval()

tqdm.tqdm.write("Loading best model: epoch %02d & %0.4f avg total loss" %
  (d["epoch"], d["total_loss"]))

## testing loop using best model
test_g_loss_ = 0
test_d_loss_ = 0
test_i_loss_ = 0

with torch.no_grad():
  for imgs, _ in test_dl:
    bsz = imgs.shape[0]
    imgs = imgs.to(DEVICE)
    real = torch.ones(bsz, 1, device=DEVICE)
    fake = torch.zeros(bsz, 1, device=DEVICE)

    ## generator loss
    gen_in = generate_noise_and_codes(NOISE_DIM, bsz, DATA, DEVICE)
    generated_imgs = generator(torch.cat(gen_in, dim=-1))
    g_pred, code_pred = discriminator(generated_imgs)
    g_loss = discriminating_loss(g_pred, real)

    ## discriminator loss
    r_pred, _ = discriminator(imgs)
    r_loss = discriminating_loss(r_pred, real)
    f_loss = discriminating_loss(g_pred, fake)
    d_loss = (r_loss + f_loss) / 2

    ## information loss
    info_loss = 0
    for ct, cp, l in zip(gen_in[1:], code_pred, code_loss):
      if isinstance(l, nn.CrossEntropyLoss): _, ct = ct.max(1)
      info_loss += l(cp, ct)
    
    ## accumulate loss
    test_g_loss_ += g_loss.item() * bsz
    test_d_loss_ += d_loss.item() * bsz
    test_i_loss_ += info_loss.item() * bsz
  
  ## take the average and log
  avg_test_g_loss = test_g_loss_ / test_size
  avg_test_d_loss = test_d_loss_ / test_size
  avg_test_i_loss = test_i_loss_ / test_size
  avg_test_t_loss = avg_test_g_loss + avg_test_d_loss + avg_test_i_loss

tqdm.tqdm.write("Test :: GLoss %0.4f :: DLoss %0.4f :: ILoss %0.4f :: TotalAvg %0.4f" %
    (avg_test_g_loss, avg_test_d_loss, avg_test_i_loss, avg_test_t_loss))

## log losses to csv
with open(LOG_DIR + "log.csv", "w") as f:
  writer = csv.writer(f)
  writer.writerow(['Generator Loss', 'Discriminator Loss', 'Information Loss', 'Total Loss'])
  for gl, dl, il, tl in zip(g_loss_epochs, d_loss_epochs, i_loss_epochs, t_loss_epochs):
    writer.writerow(('%0.5f %0.5f %0.5f %0.5f' % (gl, dl, il, tl)).split())
  writer.writerow(('%0.5f %0.5f %0.5f %0.5f' % 
    (avg_test_g_loss, avg_test_d_loss, avg_test_i_loss, avg_test_t_loss)).split())

## ------------------ GENERATING IMAGE TEST ------------------

gen_in = generate_noise_and_codes(NOISE_DIM, 10, DATA, DEVICE)
generated_imgs = generator(torch.cat(gen_in, dim=-1))
generated_imgs = (generated_imgs + 1) / 2

## change shape and convert to numpy to be visualized
if DATA == "mnist":
  generated_imgs = generated_imgs.squeeze(1)
  cmap = 'gray'

elif DATA == "celeba":
  generated_imgs = generated_imgs.moveaxis(1, -1)
  cmap = None

generated_imgs = generated_imgs.detach().cpu().numpy()

## plot
n_row, n_col = 2, 5
f, axs = plt.subplots(n_row, n_col, figsize=(n_col*5, n_row*6))
for j in range(10):
  axs[j // n_col, j % n_col].imshow(generated_imgs[j], cmap=cmap)
  axs[j // n_col, j % n_col].axis('off')
  
f.savefig(LOG_DIR + "reconstruction.png")
plt.close(f)