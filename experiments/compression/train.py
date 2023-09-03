import torch
from torchvision import utils as vutils
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import os
import json
import shutil

from experiments.utils import AverageMeter
from experiments.compression.loss import loss_function
from primis.data import ROOT_DIR, ImagePlain
from primis.models import BankEncoder, BankDecoder, IndexAutoEncoder

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-c', default='./config_train.json')

config_path = parser.parse_args().c

with open(config_path, 'r') as json_file:
    config = json.load(json_file)

train_path = os.path.join(ROOT_DIR, config["data"]["train_path"])
valid_path = os.path.join(ROOT_DIR, config["data"]["valid_path"])

writer = SummaryWriter(comment='_train')
shutil.copy('config_train.json', os.path.join(writer.log_dir, 'used_config.json'))

dataset_train = ImagePlain(split_path=train_path,
                           im_size=config["images"]["image_size"], patch_size=config["images"]["patch_size"])
dataset_valid = ImagePlain(split_path=valid_path,
                           im_size=config["images"]["image_size"], patch_size=config["images"]["patch_size"])

dataloader_train = DataLoader(dataset_train, batch_size=config["global"]["batch"], shuffle=True,
                              num_workers=config["global"]["workers"], collate_fn=dataset_train.collate_fn)
dataloader_valid = DataLoader(dataset_valid, batch_size=config["global"]["batch"], shuffle=False,
                              num_workers=config["global"]["workers"], collate_fn=dataset_valid.collate_fn)

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
loss_valid = AverageMeter('loss_valid')

best_epoch_loss_valid = float("inf")  # Just some large number
loss_epoch_valid_list = []

with torch.no_grad():
    _, _ = net(dataset_train[0]['image'].unsqueeze(0).float().to(device))

    # # writer.add_graph(net, input_to_model=dataset_train[0]['image'].unsqueeze(0).float().to(device), verbose=False)
    print('actual bits per pixel: ', net.bpp_actual)
iter_count = 0
for i_epoch in range(config["global"]["epochs"]):
    print(' **************** epoch number', i_epoch + 1, ' **********************')

    net.train()

    for i_batch, _batch in enumerate(dataloader_train):
        iter_count += 1
        inp = _batch['patches'].float().to(device)

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
            print('epoch: ({}/{}), train iter: ({}/{}),'
                  ' train mini-batch loss: {}'.format(i_epoch + 1, config["global"]["epochs"],
                                                      i_batch + 1, len(dataloader_train),
                                                      loss_train.val))

        writer.add_scalar('loss/train', loss_train.val, loss_train.count)

        if i_batch % 200 == 0:
            inp_img = patch_to_im(inp, num_channel=1, im_size=im_size)
            hat_img = patch_to_im(hat, num_channel=1, im_size=im_size)
            writer.add_image('samples/original/train', vutils.make_grid(inp_img[0:2, :, :, :]), loss_train.count)
            writer.add_image('samples/reconstructed/train', vutils.make_grid(hat_img[0:2, :, :, :]), loss_train.count)
            # writer.add_image('samples/code/train', vutils.make_grid(code[0]['pos']['ind'][0:64, 0:3, :, :]),
            #                 loss_valid.count)

        loss.backward()
        optimizer.step()

    net.eval()

    loss_batch_list_valid = []
    batch_list = []

    with torch.no_grad():
        for i_batch, _batch in enumerate(dataloader_valid):
            inp = _batch['patches'].float().to(device)
            hat, code = net(inp)
            hat = torch.clamp(hat, 0, 1)
            loss = loss_function(inp, hat)
            loss_batch_list_valid.append(loss.item())
            batch_list.append(inp.shape[0])
            if i_batch % 1 == 0:
                print('epoch: ({}/{}), valid iter: ({}/{}),'
                      ' valid mini-batch loss: {}'.format(i_epoch + 1, config["global"]["epochs"],
                                                          i_batch + 1, len(dataloader_valid),
                                                          loss.item()))

        this_epoch_loss_valid = sum([loss_batch_list_valid[_i] * batch_list[_i]
                                     for _i in range(len(batch_list))]) / sum(batch_list)

        loss_epoch_valid_list.append(this_epoch_loss_valid)
        print('Validation loss for epoch {}: {}'.format(i_epoch + 1, this_epoch_loss_valid))

        loss_valid.update(this_epoch_loss_valid)

        writer.add_scalar('loss/valid', loss_valid.val, loss_valid.count)

        inp_img = patch_to_im(inp, num_channel=1, im_size=im_size)
        hat_img = patch_to_im(hat, num_channel=1, im_size=im_size)

        writer.add_image('samples/original/valid', vutils.make_grid(inp_img[0:2, :, :, :]), loss_valid.count)
        writer.add_image('samples/reconstructed/valid', vutils.make_grid(hat_img[0:2, :, :, :]), loss_valid.count)
        # writer.add_image('samples/code/valid', vutils.make_grid(code['pos']['ind'][0:64, 0:3, :, :]),
        #                 loss_train.count)

    if best_epoch_loss_valid > this_epoch_loss_valid:
        best_epoch_loss_valid = this_epoch_loss_valid

        print('Saving the model for this epoch, as the best available..')

        torch.save(net.state_dict(), os.path.join(writer.log_dir, 'net.pth'))
    print('Best validation loss so far: {}'.format(best_epoch_loss_valid))

hparams_dict = {
    'lr': config["global"]["lr"],
    'batch': config["global"]["batch"],
    'weight_decay': config["global"]["weight_decay"],
    'dropout': config["network"]["dropout"],
    'net_params': num_params_of_net,
    'bpp_actual': net.bpp_actual,
    'bpp_ideal': net.bpp_ideal,
    'img_w': config["images"]["image_size"][0],
    'img_h': config["images"]["image_size"][1],
    'patch_w': config["images"]["patch_size"][0],
    'patch_h': config["images"]["patch_size"][1],
    "proj_filter_size": config["network"]["proj_filter_size"],
    "num_proj_channels": config["network"]["num_proj_channels"],
    "k": config["network"]["k"],
    "num_code_maps": config["network"]["num_code_maps"],
    "num_blocks": len(config["network"]["sampling_ratio_list"])
}

writer.add_hparams(hparam_dict=hparams_dict,
                   metric_dict={
                       'best_loss_valid': min(loss_epoch_valid_list),
                       'best_loss_valid_epoch': 1 + int(torch.tensor(loss_epoch_valid_list).argmin().item())
                   })

writer.close()
