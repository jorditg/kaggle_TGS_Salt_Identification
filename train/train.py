import time
import torch
import numpy as np
import glob

import torch.optim as optim
from torch.utils.data import DataLoader

from dataprocess import TGSDataset
from loss import TGSLoss
from model import LinkNet101x101

batch = 16  # 32 if you have 16 GB of VRAM
batch_per_epoch = 4000 // batch # batch per epoch
model_path = './models/resnet18_001.pth'  # Name for the model save
load_model_path = None  # Load pretrain path
encoder = 'resnet18_lite_101x101'  
final = 'sigmoid'  # Only one class, ie. sigmoid


# ISIC image normalization constants
mean = 120.34612148318782 / 255.
std = 41.069665220162044 / 255.

dataset_dir = "../input/"
train_dir = dataset_dir + "train_augmented"
val_dir = dataset_dir + "val"

np.random.seed(123)

dir1_train = sorted(glob.glob(train_dir + "/images/*.png"))
dir1_mask = sorted(glob.glob(train_dir + "/masks/*.png"))

train_image_paths = [*dir1_train]
train_target_paths = [*dir1_mask]
train_dataset = TGSDataset(train_image_paths, train_target_paths, mean, std)

val_image_paths = sorted(glob.glob(val_dir + "/images/*.png"))
val_target_paths = sorted(glob.glob(val_dir + "/masks/*.png"))
val_dataset = TGSDataset(val_image_paths, val_target_paths, mean, std)

print("Train imgs:", train_dataset.__len__())
print("Val imgs:", val_dataset.__len__())

assert torch.cuda.is_available(), "Sorry, no CUDA device found"

train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False)

device = torch.device("cuda")

train_loss = TGSLoss().to(device)
val_loss = TGSLoss().to(device)

model = LinkNet101x101(1, 1, encoder, final).to(device)
print(model)
print("Number of parameters: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
if load_model_path is not None:
    state = torch.load(load_model_path)
    model.load_state_dict(state)

def val():
    c_loss = 0
    with torch.no_grad():
        for img, trg in val_loader:
            img = img.type(torch.cuda.FloatTensor)
            trg = trg.type(torch.cuda.FloatTensor)
            pred = model(img)
            loss = val_loss(pred, trg)
            c_loss += loss.item()
        c_loss /= val_dataset.__len__()
    return c_loss

def train(epochs):
    losses = []
    best_loss = val()
    print("Start val loss:", best_loss)
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        s_time = time.time()
        for i, (img, trg) in enumerate(train_loader):
            if i > batch_per_epoch:
                break
            # get the inputs
            img = img.type(torch.cuda.FloatTensor)
            trg = trg.type(torch.cuda.FloatTensor)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            pred = model(img)
            
            loss = train_loss(pred, trg)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        running_loss /= train_dataset.__len__()
        val_s = val()
        val_s_f = round((2-val_s*batch)/2, 5)  # LB score on val
        print("Epoch:", epoch+1, "train loss:", round(running_loss, 5),
              "val loss", round(val_s, 5), "val score:", val_s_f,
              "time:", round(time.time()-s_time, 2), "s")
        if val_s < best_loss:
            torch.save(model.state_dict(), model_path[:-4] + '_cpt_' \
                       + str(val_s_f) + model_path[-4:])
            best_loss = val_s
            print("Checkpoint saved")
        losses.append([running_loss, val])
        running_loss = 0.0
    # Save the train result
    torch.save(model.state_dict(), model_path[:-4] + '_res_'\
               + str(val_s_f) + model_path[-4:])
    #print(losses)

torch.cuda.synchronize()

# Define the train protocol here
lr = 1e-4
optimizer = optim.Adam(model.parameters(), lr=lr)
train(epochs=100)
lr = 1e-5
optimizer = optim.Adam(model.parameters(), lr=lr)
train(epochs=75)
lr = 1e-6
optimizer = optim.Adam(model.parameters(), lr=lr)
train(epochs=50)

# Save the final result
torch.save(model.state_dict(), model_path)
print('Finished Training')
