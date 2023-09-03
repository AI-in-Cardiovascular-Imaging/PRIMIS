import os
import random
from data import ROOT_DIR

splits_dir = os.path.join(ROOT_DIR, 'splits')

with open(os.path.join(splits_dir, 'all.txt'), 'r') as f:
    imgs_list_all = [_l.strip() for _l in f.readlines()]

random.shuffle(imgs_list_all)

ps = [0.9, 0.05, 0.05]

imgs_list_train = imgs_list_all[0:int(ps[0]*(len(imgs_list_all)))]
imgs_list_valid = imgs_list_all[int(ps[0]*(len(imgs_list_all))): int((ps[0]+ps[1])*(len(imgs_list_all)))]
imgs_list_test = imgs_list_all[int((ps[0]+ps[1])*(len(imgs_list_all))):]

with open(os.path.join(splits_dir, 'train.txt'), 'w') as f:
    for _img in imgs_list_train:
        f.write(_img + '\n')

with open(os.path.join(splits_dir, 'valid.txt'), 'w') as f:
    for _img in imgs_list_valid:
        f.write(_img + '\n')

with open(os.path.join(splits_dir, 'test.txt'), 'w') as f:
    for _img in imgs_list_test:
        f.write(_img + '\n')

