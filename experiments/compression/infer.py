import torch
from torchvision import utils as vutils
from torch.utils.data import DataLoader

import os
import json

from primis.data import ROOT_DIR, ImagePlain, Ambiguate
from primis.models import BankEncoder, BankDecoder, IndexAutoEncoder

import argparse

# torch.manual_seed(42)

parser = argparse.ArgumentParser()

parser.add_argument('-c', default='./config_infer.json')

config_path = parser.parse_args().c

with open(config_path, 'r') as json_file:
    config = json.load(json_file)

test_path = os.path.join(ROOT_DIR, config["data"]["test_path"])

with open(os.path.join('./runs', config["train_run"], 'used_config.json'), 'r') as _json_file:
    _config = json.load(_json_file)
    config["images"] = _config["images"]
    config["network"] = _config["network"]

k = config["data"]["k"]
if k is None:
    k = config["network"]["k"]

dataset_test = ImagePlain(split_path=test_path,
                          im_size=config["images"]["image_size"], patch_size=config["images"]["patch_size"])

dataloader_test = DataLoader(dataset_test, batch_size=config["global"]["batch"], shuffle=config["global"]["shuffle"],
                             num_workers=config["global"]["workers"], collate_fn=dataset_test.collate_fn)

patch_to_im = dataset_test.patch_to_im
im_size = dataset_test.im_size

target_dir = os.path.join(config["data"]["target_dir"], config["train_run"], config["data"]["parent_dir"],
                          'k{}'.format(k))

if not os.path.exists(target_dir):
    os.makedirs(target_dir, exist_ok=True)
    os.mkdir(os.path.join(target_dir, 'original'))
    os.mkdir(os.path.join(target_dir, 'hat_sec'))
    os.mkdir(os.path.join(target_dir, 'hat_pub'))

#  Initialize the model:
device = torch.device(config["global"]["device"])
encoder = BankEncoder(config["network"]).to(device)
decoder = BankDecoder(encoder).to(device)
net = IndexAutoEncoder(encoder, decoder).to(device)

state_dict_path = os.path.join('./runs', config["train_run"], 'net.pth')
net.load_state_dict(torch.load(state_dict_path, map_location=device))
net.to(device)
net.eval()

net.set_k(k)

obj_ambiguation = Ambiguate(net)

for i_batch, _batch in enumerate(dataloader_test):
    if i_batch == 0:
        inp = _batch['patches'].float().to(device)
        with torch.no_grad():
            _, code = net(inp)
        print('Code shape: 2 x {}'.format(code['pos']['ind'].shape))
        print('Ideal rate: {}'.format(net.rate_ideal))
        print('Actual rate: {}'.format(net.rate_actual))
        print('Ideal bpp: {}'.format(net.bpp_ideal))
        print('Actual bpp: {}'.format(net.bpp_actual))

    if config["global"]["max_num_batch"] is not None:
        if i_batch >= config["global"]["max_num_batch"]:
            break
    print('batch {}/{} ................'.format(i_batch + 1, len(dataloader_test)))

    inp = _batch['patches'].float().to(device)
    names = _batch['attrb']['name']
    rel_root = _batch['attrb']['rel_root']

    # In case you want just hat_sec, uncomment the next line and comment the next 4. This will simply be faster.
    # hat_sec = patch_to_im(obj_ambiguation.decode(obj_ambiguation.encode(inp)), num_channel=1, im_size=im_size)

    res = obj_ambiguation(inp)
    orig = patch_to_im(inp, num_channel=1, im_size=im_size)
    hat_sec = patch_to_im(res['hat_sec'], num_channel=1, im_size=im_size)
    hat_pub = patch_to_im(res['hat_pub'], num_channel=1, im_size=im_size)

    for _i_img, _name in enumerate(names):
        if not os.path.exists(os.path.join(target_dir, 'original', rel_root[_i_img])):
            os.makedirs(os.path.join(target_dir, 'original', rel_root[_i_img]))
        if not os.path.exists(os.path.join(target_dir, 'hat_sec', rel_root[_i_img])):
            os.makedirs(os.path.join(target_dir, 'hat_sec', rel_root[_i_img]))
        if not os.path.exists(os.path.join(target_dir, 'hat_pub', rel_root[_i_img])):
            os.makedirs(os.path.join(target_dir, 'hat_pub', rel_root[_i_img]))

        print('Saving {}, image ({}/{}) from the current batch ..'.format(
            os.path.join(rel_root[_i_img], _name), _i_img + 1, len(names)))

        vutils.save_image(orig[_i_img], os.path.join(target_dir, 'original', rel_root[_i_img], _name))
        vutils.save_image(hat_sec[_i_img], os.path.join(target_dir, 'hat_sec', rel_root[_i_img], _name))
        vutils.save_image(hat_pub[_i_img], os.path.join(target_dir, 'hat_pub', rel_root[_i_img], _name))
