import torch
import torch.nn.functional as F
from torchvision import utils as vutils
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import sys, os, json, shutil
sys.path.insert(0, '../../')

from experiments import AverageMeter
from experiments.compression import loss_function, schedule_grads
from primis.data import ROOT_DIR, ImagePlain
from primis.models import BankEncoder, BankDecoder, IndexAutoEncoder

import argparse

# torch.manual_seed(42)

parser = argparse.ArgumentParser()

parser.add_argument('-c', default='./config_train.json')

config_path = parser.parse_args().c

with open(config_path, 'r') as json_file:
    config = json.load(json_file)

parent_dir = os.path.join(ROOT_DIR, config["data"]["parent_dir"])
train_path = os.path.join(ROOT_DIR, config["data"]["train_path"])


dataset_train = ImagePlain(parent_dir, split_path=train_path,
                           im_size=config["images"]["image_size"], patch_size=config["images"]["patch_size"])

dataloader_train = DataLoader(dataset_train, batch_size=config["global"]["batch"], shuffle=True,
                              num_workers=config["global"]["workers"], collate_fn=dataset_train.collate_fn)

patch_to_im = dataset_train.patch_to_im
im_size = dataset_train.im_size

#  Initialize the model:
device = torch.device(config["global"]["device"])

encoder = BankEncoder(config["network"]).to(device)
decoder = BankDecoder(encoder).to(device)
net = IndexAutoEncoder(encoder, decoder).to(device)

print(net.encoder.conv_channel)
print(net.encoder.conv_indexer)


def count_parameters(model):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


num_params_of_net = count_parameters(net)
print(f'The model has {num_params_of_net:,} trainable parameters')


# The optimizer
optimizer = optim.Adam(net.parameters(), lr=config["global"]["lr"], weight_decay=config["global"]["weight_decay"])


loss_train = AverageMeter('loss_train')


with torch.no_grad():
    _, _ = net(dataset_train[0]['image'].unsqueeze(0).float().to(device))

    # # writer.add_graph(net, input_to_model=dataset_train[0]['image'].unsqueeze(0).float().to(device), verbose=False)
    print('actual bits per pixel: ', net.bpp_actual)
iter_count = 0
net.train()

for i_batch, _batch in enumerate(dataloader_train):
    iter_count += 1
    inp = _batch['patches'].float().to(device)
    print(_batch['attrb'])

    hat, code = net(inp)
    if iter_count < 100:
        hat = torch.clamp(hat, -2, 2)
    else:
        hat = torch.clamp(hat, -1, 1)
    optimizer.zero_grad()
    loss = loss_function(inp, hat)
    if loss.item() > 5.0:
        print('!!!!!!!!!!!!! Something is wrong, skipping this mini-batch !!!!!')
        continue  # Just in case.
    loss_train.update(loss.item())
    if i_batch % 1 == 0:
        print('train iter: ({}/{}),'
              ' train mini-batch loss: {}'.format(
                                                  i_batch+1, len(dataloader_train),
                                                  loss_train.val))

    if i_batch % 200 == 0:
        inp_img = patch_to_im(inp, num_channel=1, im_size=im_size)
        hat_img = patch_to_im(hat, num_channel=1, im_size=im_size)
        vutils.save_image(inp_img[0:2, :, :, :], './storage/orig.png')
        vutils.save_image(hat_img[0:2, :, :, :], './storage/hat.png')

    loss.backward()
    optimizer.step()



