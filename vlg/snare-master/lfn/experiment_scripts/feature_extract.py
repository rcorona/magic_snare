# Enable import from parent package
import torch.nn.functional as F
import sys
import os
import numpy as np
import skimage.measure
import pdb
import torch.nn as nn
import json
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import multiclass_dataio
import hdf5_dataio

from collections import defaultdict
import cv2
import torch
import models
import configargparse
import config
import util
from pathlib import Path

# Get all objects that exist in SNARE. 
def get_snare_objs(split_sets=False): 
    
    snare_path = '/home/rcorona/obj_part_lang/snare-master/amt/folds_adversarial'
    train = json.load(open(os.path.join(snare_path, 'train.json')))
    val = json.load(open(os.path.join(snare_path, 'val.json')))
    test = json.load(open(os.path.join(snare_path, 'test.json')))

    train_objs = set()
    val_objs = set()
    test_objs = set()

    # Comb through snare files to collect unique set of ShapeNet objects. 
    snare_objs = set()

    for obj_set, split in [(train_objs, train), (val_objs, val), (test_objs, test)]:
        for datapoint in split: 
            for obj in datapoint['objects']:
                obj_set.add(obj)

    all_objs = train_objs | val_objs | test_objs

    if not split_sets: 
        return all_objs
    else:
        return (train_objs, val_objs, test_objs)

class InstanceDataset(): 

    def __init__(self, instances):
        self.instances = instances

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        return self.instances[idx]

def collate_fn(datapoints):
    
    result = {}

    # Simply add another dimension to everything. 
    for k in datapoints[0].keys(): 
        result[k] = torch.stack([datapoint[k] for datapoint in datapoints]).cuda()

    return result

class SNSemDataset():

    def __init__(self, path, view_idxs=[0,1,2,3,4,5,6,7]):
        self.path = path

        # List of all snare objects. 
        self.objs = list(get_snare_objs())

        sidelen = 64
        org_sidelength = 64

        # UV is the same for all images. 
        uv = np.mgrid[0:org_sidelength, 0:org_sidelength].astype(np.int32).transpose(1, 2, 0)
        uv = cv2.resize(uv, (sidelen, sidelen), interpolation=cv2.INTER_NEAREST)
        uv = torch.from_numpy(np.flip(uv, axis=-1).copy()).long()
        uv = uv.reshape(-1, 2).float()
        self.uv = uv

        # Poses are the same for the same view across objects. 
        cam_param_path = '/home/rcorona/2022/lang_nerf/vlg/snare-master/data/cameras.npy'
        cam_params = np.load(cam_param_path, allow_pickle=True).item()
        self.poses = [torch.from_numpy(cam_params['world_mat_inv_{}'.format(idx)]).float() for idx in range(8)]

        # Camera intrinsics. 
        intrinsics_path = '/home/rcorona/data/NMR_Dataset/intrinsics.txt'
        intrinsics, _, _, _ = util.parse_intrinsics(intrinsics_path,
                                            trgt_sidelength=64)
        self.intrinsics = torch.Tensor(intrinsics).float()

        # Views to provide for each object as context. 
        self.view_idxs = view_idxs

    def __len__(self):
        return len(self.objs)

    def __getitem__(self, idx): 

        # Object id. 
        obj_idx = self.objs[idx]
        img_dir = os.path.join(self.path, obj_idx)

        # Will be formed into datapoint. 
        rgb = []
        cam2world = []
        uv = []
        intrinsics = []
        instance_idx = []

        # Will contain views as individuals. 
        views = []

        # Dictionary of views for object. 
        for view in range(8):
            
            view_dict = {}

            # Name based on degree of rotation. 
            degree = "{:03d}".format(view * 45)
            img_path = os.path.join(img_dir, '{}.png'.format(degree))

            # Load image and set white background to fit PixelNeRF distribution. 
            img = Image.open(img_path).convert("RGBA")
            background = Image.new('RGBA', img.size, (255,255,255))
            alpha_composite = Image.alpha_composite(background, img)
            alpha_composite_3 = alpha_composite.convert('RGB')
            img = np.asarray(alpha_composite_3).astype(np.float32) / 255.0

            # For iterating over queries. 
            view_dict = {
                'rgb': torch.from_numpy(img.reshape(-1, 3)).float(),
                'cam2world': self.poses[view],
                'uv': self.uv,
                'intrinsics': self.intrinsics,
                'instance_idx': torch.Tensor([idx]).squeeze()
            }

            views.append(view_dict)

            # Add view data to datapoint components (but only if we want it as context)
            if view in self.view_idxs: 
                rgb.append(view_dict['rgb'])
                cam2world.append(view_dict['cam2world'])
                uv.append(view_dict['uv'])
                intrinsics.append(view_dict['intrinsics'])
                instance_idx.append(view_dict['instance_idx'])
            
        # Combine into single datapoint. 
        context = {
            'rgb': torch.stack(rgb),
            'cam2world': torch.stack(cam2world),
            'uv': torch.stack(uv),
            'intrinsics': torch.stack(intrinsics),
            'instance_idx': torch.stack(instance_idx)
        }

        return (context, InstanceDataset(views))

p = configargparse.ArgumentParser()
p.add_argument('--data_root', type=str, required=True)
p.add_argument('--dataset', type=str, required=True)
p.add_argument('--checkpoint_path', required=True)
p.add_argument('--network', type=str, default='relu')
p.add_argument('--conditioning', type=str, default='hyper')
p.add_argument('--max_num_instances', type=int, default=None)
p.add_argument('--img_sidelength', type=int, default=64, required=False)
p.add_argument('--viewlist', type=str, default=None, required=False)

opt = p.parse_args()

state_dict = torch.load(opt.checkpoint_path)
num_instances = state_dict['latent_codes.weight'].shape[0]

model = models.LFAutoDecoder(num_instances=num_instances, latent_dim=256, parameterization='plucker', network=opt.network,
                             conditioning=opt.conditioning)
model.eval()
print("Loading model")
model.load_state_dict(state_dict)

# Overwrite latent codes. 
model.latent_codes = nn.Embedding(model.num_instances, model.latent_dim)
nn.init.normal_(model.latent_codes.weight, mean=0, std=0.01)
model = model.cuda()

def convert_image(img, type):
    img = img[0]

    if not 'normal' in type:
        img = util.lin2img(img)[0]
    img = img.cpu().numpy().transpose(1, 2, 0)

    if 'rgb' in type or 'normal' in type:
        img += 1.
        img /= 2.
    elif type == 'depth':
        img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
    img *= 255.
    img = np.clip(img, 0., 255.).astype(np.uint8)
    return img

print("Loading dataset")
if opt.dataset == 'NMR':
    dataset = multiclass_dataio.get_instance_datasets(opt.data_root, sidelen=opt.img_sidelength, dataset_type='test',
                                                      max_num_instances=opt.max_num_instances)

elif opt.dataset == 'SNSem': 
    dataset = SNSemDataset(opt.data_root)
else:
    raise


# Select random item from the dataset. 
idx = np.random.randint(len(dataset))
print(idx)

# For optimizing parameters. 
optim = torch.optim.Adam(model.latent_codes.parameters(), lr=1e-3)

# Training query. 
context, queries = dataset[idx]
model_input = {'query': collate_fn([context])}

# Training loop. 
for i in range(1000): 

    # Forward pass through model. 
    model_output = model(model_input)
    pred = model_output['rgb']
    gt = model_input['query']['rgb']

    # Loss computation. 
    optim.zero_grad()
    loss = nn.MSELoss()(gt, pred) * 200
    loss.backward()
    optim.step()

    print('Loss: {}'.format(loss))

with torch.no_grad():
    for j, query in enumerate(queries):

        model_input = util.assemble_model_input(query, query)
        model_output = model(model_input)

        out_dict = {}
        out_dict['rgb'] = model_output['rgb']
        out_dict['gt_rgb'] = model_input['query']['rgb']

        # Ground truth and predicted images. 
        gt_img = convert_image(out_dict['gt_rgb'], 'rgb')
        img = convert_image(out_dict['rgb'], 'rgb')

        # Write to test folder. 
        out_dir = 'test_imgs'
        cv2.imwrite(os.path.join(out_dir, '{}.png'.format(j)), img)
        cv2.imwrite(os.path.join(out_dir, '{}_gt.png'.format(j)), gt_img)