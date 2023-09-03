import torch
import torch.nn.functional as F

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms, utils

ROOT_DIR = "/path/to/your/database"


class ImagePlain(Dataset):
    """
    The main torch-style dataset class. Reads images from source, with a list specified by split path.
    If patch-based processing is desired, it will happen when used with a dataloader and the specified collate function.

    TODO: Provide functionality for identifying patch orders, for possible position encoding in the future.
    """

    def __init__(self, split_path, im_size, transform=None, patch_size=None):
        """
        :param split_path: a text file listing the relative paths of images of interest w.r.t. the ROOT_DIR
        :param im_size: (H,W)
        :param transform: a torchvision thing
        """

        self.root_dir = ROOT_DIR
        self.split_path = split_path
        self.im_size = im_size
        self.transform = transform
        self.patch_size = patch_size
        with open(self.split_path, 'r') as f:
            lines = [_l.strip() for _l in f.readlines()]
        self.image_names = [os.path.split(_l)[-1] for _l in lines]
        self.image_rel_roots = [os.path.join(*os.path.split(_l)[0:-1]) for _l in lines]

    @staticmethod
    def im_to_patch(im, patch_size):
        """
        A simple interface to patch-order 4D image tensors

        :param im:
        :param patch_size:
        :return:
        """
        if type(patch_size) is not tuple:
            if type(patch_size) is list:
                patch_size = tuple(patch_size)
            else:
                patch_size = (patch_size, patch_size)

        im_size = (im.size()[2], im.size()[3])
        im_size_ext = im_size
        if im_size[0] % patch_size[0] != 0 or im_size[1] % patch_size[1]:
            pad = (0, patch_size[1] - (im_size[1] % patch_size[1]), 0, (patch_size[0] - im_size[0] % patch_size[0]))
            im = F.pad(im, pad, 'reflect')

        patch = im.unfold(2, patch_size[0], patch_size[0]).unfold(3, patch_size[1], patch_size[1])
        return patch.permute(0, 2, 3, 1, 4, 5).reshape(-1, im.size()[1], patch_size[0], patch_size[1])

    @staticmethod
    def patch_to_im(patch, num_channel, im_size):
        """
        A simple interface to reconstruct a 4D image tensor from its patches
        :param patch: The patches as a 4D tensor.
        :param num_channel:
        :param im_size:
        :return:
        """
        if type(im_size) is not tuple:
            if type(im_size) is list:
                im_size = tuple(im_size)
            else:
                im_size = (im_size, im_size)

        patch_size = (patch.size()[2], patch.size()[3])
        #
        im_size_exp = list(im_size)
        if im_size[0] % patch_size[0] != 0:
            im_size_exp[0] = im_size[0] + patch_size[0] - im_size[0] % patch_size[0]
        if im_size[1] % patch_size[1] != 0:
            im_size_exp[1] = im_size[1] + patch_size[1] - im_size[1] % patch_size[1]

        num_patches = (int(im_size_exp[0] / patch_size[0]), int(im_size_exp[1] / patch_size[1]))
        patch = patch.permute(1, 0, 2, 3).view(-1, num_channel, num_patches[0], num_patches[1], patch_size[0],
                                               patch_size[1])
        im = patch.permute(0, 1, 2, 4, 3, 5)
        im = im.contiguous().view(-1, num_channel, im_size_exp[0], im_size_exp[1])
        return im[:, :, 0:im_size[0], 0:im_size[1]]

    def collate_fn(self, inp):
        out = {}
        images = torch.cat(tuple([_sample['image'].unsqueeze(0) for _sample in inp]), dim=0)
        out['attrb'] = {'name': [_sample['attrb']['name'] for _sample in inp],
                        'rel_root': [_sample['attrb']['rel_root'] for _sample in inp]}

        if not self.patch_size:
            out['image'] = images

            return out
        out['patches'] = self.im_to_patch(images, patch_size=self.patch_size)

        return out

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        _name = self.image_names[idx]
        _root = self.image_rel_roots[idx]
        image_path = os.path.realpath(os.path.join(self.root_dir, _root, _name))
        image = Image.open(image_path)
        image = image.resize(self.im_size, Image.NEAREST)
        if self.transform:
            image = self.transform(image)
        image = transforms.ToTensor()(image)

        sample = {
            'image': image[0].unsqueeze(0),  # Weired that some images show up as having 4 channels instead of 1.
            'attrb': {
                'name': _name,
                'rel_root': _root
            },
        }

        return sample


class Ambiguate:
    """
    Ambiguation and dis-ambiguation mechanism.
    """

    def __init__(self, net):
        self.net = net
        self.net.eval()
        self.device = next(self.net.parameters()).device

        self.k = self.net.encoder.k
        self.m = self.net.encoder.m
        self.dim = self.net.encoder.c_o

    def encode(self, inp):
        inp = inp.to(self.device)
        with torch.no_grad():
            code = self.net.encoder(inp)
        return code

    def ambiguate(self, code):
        ind_pos = code["pos"]["ind"]
        ind_neg = code["neg"]["ind"]
        val_pos = code["pos"]["val"]
        val_neg = code["neg"]["val"]

        _b, _k, _m, _w, _h = val_pos.shape
        assert _k == self.k and _m == self.m

        min_pos = val_pos.min().item()
        max_pos = val_pos.max().item()
        min_neg = val_neg.min().item()
        max_neg = val_neg.max().item()

        pub_pos = (max_pos - min_pos) * torch.rand(int(_b * self.dim * _m * _w * _h / 2), device=self.device) + max_pos
        pub_neg = (max_neg - min_neg) * torch.rand_like(pub_pos) + max_neg

        pub = torch.cat((torch.zeros_like(pub_pos), torch.zeros_like(pub_neg)), 0)
        perm = torch.randperm(pub.numel())
        pub[perm[0:int(pub.numel() / 2)]] = pub_pos
        pub[perm[int(pub.numel() / 2): pub.numel()]] = pub_neg

        pub = pub.view(_b, self.dim, _m, _w, _h)
        pub.scatter_(1, ind_pos, val_pos)
        pub.scatter_(1, ind_neg, val_neg)

        key = {'pos': ind_pos, 'neg': ind_neg}

        return pub, key

    def do(self, inp):
        code = self.encode(inp)
        pub, key = self.ambiguate(code)

        return pub, key

    def disambiguate(self, pub, key=None):
        _b, _d, _m, _w, _h = pub.shape
        assert _d == self.dim and _m == self.m
        code = {
            'pos': {'ind': None, 'val': None},
            'neg': {'ind': None, 'val': None}
        }

        if key is not None:
            code['pos']['ind'] = key['pos']
            code['neg']['ind'] = key['neg']

            code['pos']['val'] = torch.gather(pub, 1, key['pos'])
            code['neg']['val'] = torch.gather(pub, 1, key['neg'])

            return code

        # code['pos']['ind'] = torch.arange(self.dim).view(1, -1, 1, 1, 1).repeat(_b, 1, _m, _w, _h)

        code['pos']['val'], code['pos']['ind'] = torch.topk(pub, self.k, dim=1)
        code['neg']['val'], code['neg']['ind'] = torch.topk(-pub, self.k, dim=1)

        return code

    def decode(self, code):
        with torch.no_grad():
            hat = self.net.decoder(code)

        return hat

    def undo(self, pub, key=None):
        code = self.disambiguate(pub, key=key)
        hat = self.decode(code)

        return hat

    def __call__(self, inp):
        pub, key = self.do(inp)
        hat_sec = self.undo(pub, key=key)
        hat_pub = self.undo(pub, key=None)

        res = {'pub': pub, 'key': key,
               'hat_sec': hat_sec, 'hat_pub': hat_pub}

        return res
