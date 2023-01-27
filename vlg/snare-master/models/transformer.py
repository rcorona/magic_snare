import numpy as np
import json
import os
from pathlib import Path
import wandb
import pdb
import math
import cv2
from functools import wraps
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
import clip
from torchvision.utils import make_grid
from einops import rearrange, repeat, reduce
from torch import einsum

from legoformer import LegoFormerM, LegoFormerS
from legoformer.util.utils import load_config
import models.aggregator as agg
from legoformer.model.transformer import Encoder
from legoformer.util.metrics import calculate_iou, calculate_fscore
from data.peract_voxelizer import VoxelGrid

## From https://github.com/peract/peract/blob/main/agents/peract_bc/perceiver_lang_io.py
LRELU_SLOPE = 0.02

def act_layer(act):
    if act == 'relu':
        return nn.ReLU()
    elif act == 'lrelu':
        return nn.LeakyReLU(LRELU_SLOPE)
    elif act == 'elu':
        return nn.ELU()
    elif act == 'tanh':
        return nn.Tanh()
    elif act == 'prelu':
        return nn.PReLU()
    else:
        raise ValueError('%s not recognized.' % act)

class Conv3DBlock(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_sizes: Union[int, list]=3, strides=1,
                 norm=None, activation=None, padding_mode='replicate',
                 padding=None):
        super(Conv3DBlock, self).__init__()
        padding = kernel_sizes // 2 if padding is None else padding
        self.conv3d = nn.Conv3d(
            in_channels, out_channels, kernel_sizes, strides, padding=padding,
            padding_mode=padding_mode)

        if activation is None:
            nn.init.xavier_uniform_(self.conv3d.weight,
                                    gain=nn.init.calculate_gain('linear'))
            nn.init.zeros_(self.conv3d.bias)
        elif activation == 'tanh':
            nn.init.xavier_uniform_(self.conv3d.weight,
                                    gain=nn.init.calculate_gain('tanh'))
            nn.init.zeros_(self.conv3d.bias)
        elif activation == 'lrelu':
            nn.init.kaiming_uniform_(self.conv3d.weight, a=LRELU_SLOPE,
                                     nonlinearity='leaky_relu')
            nn.init.zeros_(self.conv3d.bias)
        elif activation == 'relu':
            nn.init.kaiming_uniform_(self.conv3d.weight, nonlinearity='relu')
            nn.init.zeros_(self.conv3d.bias)
        else:
            raise ValueError()

        self.activation = None
        self.norm = None
        if norm is not None:
            raise NotImplementedError('Norm not implemented.')
        if activation is not None:
            self.activation = act_layer(activation)
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv3d(x)
        x = self.norm(x) if self.norm is not None else x
        x = self.activation(x) if self.activation is not None else x
        return x

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = None

    @wraps(f)
    def cached_fn(*args, _cache=True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache

    return cached_fn

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)
    
class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        # dropout
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)
##

## From https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
        # Learned embeddings for text vs. voxel tokens. 
        self.text_token = nn.Parameter(torch.normal(torch.zeros((1,1,d_model)), torch.full((1,1,d_model), 0.1)))
        self.voxel_token = nn.Parameter(torch.normal(torch.zeros((1,1,d_model)), torch.full((1,1,d_model), 0.1)))

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        return self.pe[:x.size(0)]
##

class TransformerClassifier(LightningModule):

    def __init__(self, cfg, train_ds=None, val_ds=None):
        self.optimizer = None
        super().__init__()

        self.cfg = cfg
        self.dropout = self.cfg['train']['dropout']

        # Keep track of predictions for visualizations later. 
        self.val_predictions = {
            'probs': [],
            'labels': [],
            'visual': []
        }

        # Determines the modalities used by the model. 
        self.feats_backbone = self.cfg['train']['feats_backbone']

        if self.feats_backbone == 'clip' or self.feats_backbone == 'multimodal': 
            self.use_imgs = True
        else: 
            self.use_imgs = True

        # Fine-tuned or frozen Legoformer/CLIP
        self.frozen_legoformer = self.cfg['transformer']['freeze_legoformer']
        self.frozen_clip = self.cfg['transformer']['freeze_clip']

        # Constants
        self.img_feat_dim = 512
        self.lang_feat_dim = 512
        self.feat_dim = 256
        self.num_views = 8
        
        if self.cfg['data']['use_rgb_pc']:
            self.feat_dim = 64

        # Use perceiver for explicit voxel maps provide directly or through point clouds. 
        self.use_peract = self.cfg['data']['use_explicit_voxels'] or self.cfg['data']['use_rgb_pc']

        # For RGB point clouds. 
        if self.cfg['data']['use_rgb_pc']:
            
            self.bounds = torch.Tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]).cuda()
            
            self.voxelizer = VoxelGrid(coord_bounds=self.bounds,
                voxel_size=32,
                device='cuda:0',
                batch_size=self.cfg['train']['batch_size'],
                feature_size=3,
                max_num_coords=8912,
            ).cuda()
            
            init_dim = 10
            im_channels = 64
            voxel_patch_size = 5
            voxel_patch_stride = 5
            activation = 'relu'
            
            self.input_preprocess = Conv3DBlock(
                init_dim, im_channels, kernel_sizes=1, strides=1,
                norm=None, activation=activation
            ).cuda()
            
            self.patchify = Conv3DBlock(
                self.input_preprocess.out_channels, im_channels,
                kernel_sizes=voxel_patch_size, strides=voxel_patch_stride,
                norm=None, activation=activation
            ).cuda()

        # 3D CNN for explicit voxelmaps. 
        if self.cfg['data']['use_explicit_voxels']:
            
            # Patch feature extractor. 
            self.conv3d = nn.Sequential(
                nn.Conv3d(in_channels=1,out_channels=32, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3),
                nn.ReLU(),
                nn.MaxPool3d(2),
                nn.Dropout(p=0.3)
            )
            
        if self.use_peract:
            
            # Perceiver parameters. 
            cross_heads = 1
            cross_dim_head = 64
            input_dropout = 0.1
            input_dim_before_seq = self.feat_dim
            depth = 6 
            weight_tie_layers = False
            latent_dim = self.feat_dim
            latent_heads = 8
            latent_dim_head = 64
            attn_dropout = 0.1
            decoder_dropout = 0.0    
            lang_max_seq_len = 100

            if self.cfg['data']['use_explicit_voxels']:
                spatial_size = 6
            elif self.cfg['data']['use_rgb_pc']:
                spatial_size = 20
        
            # Perciever latents. 
            n_latents = self.cfg['transformer']['n_latents']
            self.latents = nn.Parameter(torch.randn(n_latents, input_dim_before_seq))

            # Learned positional encodings for perceiver. 
            self.pos_encoding = nn.Parameter(torch.randn(1,
                                                         lang_max_seq_len + spatial_size ** 3 + 8,
                                                         input_dim_before_seq))
            
            # Perceiver. 
            self.cross_attend_blocks = nn.ModuleList([
                PreNorm(self.feat_dim, Attention(self.feat_dim,
                                            input_dim_before_seq,
                                            heads=cross_heads,
                                            dim_head=cross_dim_head,
                                            dropout=input_dropout),
                        context_dim=input_dim_before_seq),
                PreNorm(self.feat_dim, FeedForward(self.feat_dim))
            ])
            
            self.layers = nn.ModuleList([])
            cache_args = {'_cache': weight_tie_layers}

            get_latent_attn = lambda: PreNorm(latent_dim,
                                            Attention(latent_dim, heads=latent_heads,
                                                        dim_head=latent_dim_head, dropout=attn_dropout))
            get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim))
            get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

            for i in range(depth):
                self.layers.append(nn.ModuleList([
                    get_latent_attn(**cache_args),
                    get_latent_ff(**cache_args)
                ]))

            # decoder cross attention
            self.decoder_cross_attn = PreNorm(input_dim_before_seq, Attention(input_dim_before_seq,
                                                                                latent_dim,
                                                                                heads=cross_heads,
                                                                                dim_head=cross_dim_head,
                                                                                dropout=decoder_dropout),
                                            context_dim=latent_dim)

        # Determine dimension of object features. 
        if self.feats_backbone == 'legoformer': 
            
            if self.cfg['data']['use_explicit_voxels']:
                self.obj_feat_dim = 32
            else: 
                if self.cfg['transformer']['xyz_embeddings']:
                    self.obj_feat_dim = 32 * 3 
                else: 
                    self.obj_feat_dim = 768

        elif self.feats_backbone == 'pix2vox': 
            self.obj_feat_dim = 8192

        # Bypass this in case we want to directly use VGG16 embeddings. 
        if self.cfg['transformer']['skip_legoformer']:            
            self.obj_feat_dim = 4096

        print('Using obj_feat_dim: {}'.format(self.obj_feat_dim))

        # build network
        self.build_model()

        # Used to keep track of train progress. 
        self.step_num = 0
        self.val_step_num = 0
        self.epoch_num = 0
        self.log_dict = {'step_num': 0}

        # val progress
        self.best_val_acc = -1.0
        self.best_val_res = None

        # test progress
        self.best_test_acc = -1.0
        self.best_test_res = None

        # results save path
        self.save_path = Path(os.path.join(os.getcwd(), 'checkpoints'))

        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)

        print('Checkpoint path: {}'.format(self.save_path))

        # log with wandb
        self.log_data = self.cfg['train']['log']
        if self.log_data:
            self.run = wandb.init(
                project=self.cfg['wandb']['logger']['project'],
                config=self.cfg['train'],
                settings=wandb.Settings(show_emoji=False),
                reinit=True
            )
            wandb.run.name = self.cfg['train']['exp_name']
            
    def build_model(self):
        
        # Use legoformer to fine-tune if not freezing. 
        if not self.frozen_legoformer:
            # Determine if single or multiview legoformer. 
            if self.cfg['data']['n_views'] == 1:
                legoformer_class = LegoFormerS
                ckpt_path = self.cfg['legoformer_paths']['legoformer_s']
                cfg_path = os.path.join(self.cfg['legoformer_paths']['cfg'], 'legoformer_s.yaml')
            else:
                legoformer_class = LegoFormerM
                ckpt_path = self.cfg['legoformer_paths']['legoformer_m']
                cfg_path = os.path.join(self.cfg['legoformer_paths']['cfg'], 'legoformer_m.yaml')

            # Load pre-trained legoformer. 
            cfg = load_config(cfg_path)
            self.legoformer = legoformer_class.load_from_checkpoint(ckpt_path, config=cfg)

        # CLIP-based langauge model. Frozen. # TODO Do we want to add option to fine-tune?  
        self.clip, _ = clip.load('ViT-B/32', device='cuda')

        if self.frozen_clip: 
            for p in self.clip.parameters():
                p.requires_grad = False

        # choose aggregation method
        agg_cfg = dict(self.cfg['train']['aggregator'])
        agg_cfg['input_dim'] = self.img_feat_dim
        self.aggregator_type = self.cfg['train']['aggregator']['type']
        self.aggregator = agg.names[self.aggregator_type](agg_cfg)

        # image encoder
        if self.use_imgs:  
            self.img_fc = nn.Sequential(
                nn.Linear(self.img_feat_dim, self.feat_dim), 
                nn.GELU(), 
                nn.LayerNorm(self.feat_dim)
            )

        # language encoder
        self.lang_fc = nn.Sequential(
            nn.Linear(self.lang_feat_dim, self.feat_dim), 
            nn.GELU(),
            nn.LayerNorm(self.feat_dim)
        )

        # Object encoder.
        self.obj_fc = nn.Sequential(
            nn.Linear(self.obj_feat_dim, self.feat_dim), 
            nn.GELU(),
            nn.LayerNorm(self.feat_dim)
        )

        # Transformer layers over modalities. 
        self.transformer = Encoder(self.feat_dim, filter_size=self.feat_dim, n_head=8, dropout=self.dropout, 
                n_layers=self.cfg['transformer']['layers'], pre_lnorm=True)

        # Positional encoding for transformer. 
        self.positional_encoding = PositionalEncoding(self.feat_dim, self.dropout)

        # Classification token. 
        #self.cls_token = nn.Parameter(torch.normal(0.0, 1.0, (1, self.feat_dim)))
        self.cls_token = nn.Parameter(torch.normal(0.0, 1.0, (1, self.feat_dim)))

        ## TODO remove
        self.transformer_mlp = nn.Sequential(nn.Linear(512 * 4, 512), nn.ReLU())
        self.obj1_token = torch.normal(0.0, 1.0, (1, self.feat_dim)).to('cuda')
        self.obj2_token = torch.normal(0.0, 1.0, (1, self.feat_dim)).to('cuda')
       
        # Vision & Language stream. 
        self.vl_mlp = nn.Sequential(
            nn.Linear(self.lang_feat_dim + self.img_feat_dim, 512), 
            nn.ReLU(True), 
            nn.Dropout(self.dropout), 
            nn.Linear(512, self.feat_dim), 
            nn.ReLU(True)
        )

        # Used instead of transformer head. 
        self.mlp = nn.Sequential(
            nn.Linear(self.feat_dim * 2, self.feat_dim * 2),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
        )

        ##

        # CLS layers for classification
        cls_in_dim = self.feat_dim * 2 if not self.cfg['transformer']['skip_clip'] else self.feat_dim

        self.cls_fc = nn.Sequential(
            nn.Linear(cls_in_dim, self.feat_dim // 2),
            nn.ReLU(True), # TODO GeLU
            nn.Dropout(self.dropout),
            nn.Linear(self.feat_dim // 2, 1)
        )

    def configure_optimizers(self):
        params_to_optimize = [p for p in self.parameters() if p.requires_grad]
        # TODO wd = 1e-3, 1e-4, 0.01, 0.05
        # TODO 1e-3
        # TODO Big model that is regularized. 

        if self.cfg['transformer']['optim'] == 'adam':
            self.optimizer = torch.optim.Adam(params_to_optimize, lr=self.cfg['transformer']['lr'], weight_decay=self.cfg['train']['weight_decay'])

        elif self.cfg['transformer']['optim'] == 'adamW':
            self.optimizer = torch.optim.AdamW(params_to_optimize, lr=self.cfg['transformer']['lr'], weight_decay=self.cfg['train']['weight_decay'])


        # Linear scheduler. 
        def linear_warmup(step): 
            return min(step / self.cfg['transformer']['warmup_steps'], 1.0)

        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, linear_warmup)
        scheduler_cfg = {
                'scheduler': scheduler, 
                'interval': 'step', 
                'frequency': 1
        }

        return ([self.optimizer], [scheduler_cfg])

    def smoothed_cross_entropy(self, pred, target, alpha=0.1):
        # From ShapeGlot (Achlioptas et. al)
        # https://github.com/optas/shapeglot/blob/master/shapeglot/models/neural_utils.py
        n_class = pred.size(1)
        one_hot = target
        one_hot = one_hot * ((1.0 - alpha) + alpha / n_class) + (1.0 - one_hot) * alpha / n_class  # smoothed
        log_prb = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).sum(dim=1)
        return torch.mean(loss)

    def _criterion(self, out):
        probs = out['probs']
        labels = out['labels']

        ce_loss = self.smoothed_cross_entropy(probs, labels)

        return_dict = {'ce_loss': ce_loss}

        # Additionally use volumetric reconstruction loss. 
        if self.cfg['train']['reconstruction_loss']:

            obj1_map_pred, obj2_map_pred = out['reconstructions']
            obj1_map_gt, obj2_map_gt = out['gt_voxels']
            vmask = torch.cat(out['voxel_masks'])

            # Compute reconstruction loss for each object voxel map. 
            reconstruction_loss1 = self.legoformer.calculate_loss(obj1_map_pred, obj1_map_gt, reduction='none')
            reconstruction_loss2 = self.legoformer.calculate_loss(obj2_map_pred, obj2_map_gt, reduction='none') 
            reconstruction_loss = torch.cat([reconstruction_loss1, reconstruction_loss2], dim=0)

            # Mask out any invalid voxels.
            reconstruction_loss = reconstruction_loss.view(reconstruction_loss.size(0), -1).sum(-1) * vmask
            reconstruction_loss = reconstruction_loss.sum() / vmask.sum()

            # Weighted sum. # TODO should we tune weight? 
            lmbda = int(self.cfg['train']['loss_lambda'])
            loss = lmbda * ce_loss + (1.0 - lmbda) * reconstruction_loss

            return_dict['loss'] = loss
            return_dict['reconstruction_loss']: reconstruction_loss
        else: 
            return_dict['loss'] = ce_loss

        return return_dict

    def transformer_pass(self, feats, padding_mask, lang_length, get_weights=False): 
        feats = feats.permute(1, 0, 2)
        
        # Get positional encoding, but assign same "position" to object tokens since order doesn't matter. 
        positional_encoding = self.positional_encoding(feats)
        
        # Add language positions. 
        feats[:lang_length] = feats[:lang_length] + positional_encoding[:lang_length]
        feats[:lang_length] = feats[:lang_length] + self.positional_encoding.text_token.expand_as(feats[:lang_length])
        feats[:lang_length] = self.positional_encoding.dropout(feats[:lang_length])
        
        # Add object token positions if using explicit voxelmap. 
        if self.cfg['data']['use_explicit_voxels']:
            
            # Add text positional encoding and dropout. 
            feats[lang_length:-1] = feats[lang_length:-1] + positional_encoding[:len(feats[lang_length:-1])]
            feats[lang_length:-1] = feats[lang_length:-1] + self.positional_encoding.voxel_token.expand_as(feats[lang_length:-1])
            feats[lang_length:-1] = self.positional_encoding.dropout(feats[lang_length:-1])

        # Pass tokens through transformer. 
        if get_weights: 
            feats, attn_weights = self.transformer.visualization_forward(feats, padding_mask)
        else: 
            feats = self.transformer(feats, padding_mask)

        feats = feats.permute(1, 0, 2)
        feats = feats[:,-1]

        if get_weights: 
            return (feats, attn_weights)
        else: 
            return feats

    def patchify_rgbpc(self, rgbpc):
        bz = rgbpc.size(0)
        
        pcd = rgbpc[:,:,:3]
        rgb = rgbpc[:,:,3:]
        
        # Voxelize point cloud and patchify. 
        voxel_grid = self.voxelizer.coords_to_bounding_voxel_grid(
            pcd, coord_features=rgb, coord_bounds=self.bounds)
        voxel_grid = voxel_grid.permute(0, 4, 1, 2, 3).detach()
        
        d0 = self.input_preprocess(voxel_grid)
        ins = self.patchify(d0)
        
        return rearrange(ins, 'b d ... -> b (...) d')

    def forward(self, batch):
        
        # Unpack features.  
        img_feats = batch['img_feats'] if 'img_feats' in batch else None        
        obj_feats = batch['obj_feats'] if 'obj_feats' in batch else None
        imgs = batch['images'] if 'images' in batch else None
        vgg16_feats = batch['vgg16_feats'] if 'vgg16_feats' in batch else None
        lang_tokens = batch['lang_tokens'].cuda()
        voxel_maps = batch['voxel_maps'] if 'voxel_maps' in batch else None
        voxel_masks= batch['voxel_masks'] if 'voxel_masks' in batch else None

        ans = batch['ans'].cuda()
        (key1, key2) =  batch['keys']
        annotation = batch['annotation']
        is_visual = batch['is_visual']

        # TODO do we want to feed all of these into transformer, or just the aggregate? 
        # Load, aggregate, and process img features. 
        if img_feats: 
            img1_n_feats = img_feats[0].to(device=self.device).float()
            img2_n_feats = img_feats[1].to(device=self.device).float()  

            img1_feats = self.aggregator(img1_n_feats)
            img2_feats = self.aggregator(img2_n_feats)

            # Project into shared embedding space. 
            #img1_feats = self.img_fc(img1_feats)
            #img2_feats = self.img_fc(img2_feats)

        # Generate object features using legoformer.  
        # Right now we assume we've precomputed the VGG16 features and don't use raw images. 
        if self.cfg['train']['feats_backbone'] == 'legoformer':
            
            # If frozen legoformer, then just use pre-extracted features. 
            if self.frozen_legoformer: 
                obj1_n_feats, obj2_n_feats = obj_feats
                obj1_n_feats = obj1_n_feats.cuda()
                obj2_n_feats = obj2_n_feats.cuda()
                
                obj1_reconstruction, obj2_reconstruction = None, None
                
                # Post process with 3D CNN if using explicit voxelmap. 
                if self.cfg['data']['use_explicit_voxels']:
                     
                    bz = obj1_n_feats.size(0)
                    
                    # Extract patch level features.  
                    obj1_n_feats = self.conv3d(obj1_n_feats.unsqueeze(1))
                    obj2_n_feats = self.conv3d(obj2_n_feats.unsqueeze(1))
                    
                    # Rearrange for inputting into perceiver. 
                    obj1_n_feats = rearrange(obj1_n_feats, 'b d ... -> b (...) d')
                    obj2_n_feats = rearrange(obj2_n_feats, 'b d ... -> b (...) d')
                
                elif self.cfg['data']['use_rgb_pc']:

                    obj1_n_feats = self.patchify_rgbpc(obj1_n_feats)
                    obj2_n_feats = self.patchify_rgbpc(obj2_n_feats)
                    
            # Otherwise extract them. 
            else: 
                vgg16_feats1, vgg16_feats2 = vgg16_feats
                vgg16_feats1, vgg16_feats2 = vgg16_feats1.cuda(), vgg16_feats2.cuda()

                # Potentially skip legoformer all together and use VGG16 features directly. 
                if not self.cfg['transformer']['skip_legoformer']:
                    # Also optionally get reconstruction output.
                    reconstruction = self.cfg['data']['voxel_reconstruction']
                    xyz_feats = self.cfg['transformer']['xyz_embeddings']
                    obj1_n_feats, obj1_reconstruction = self.legoformer.get_obj_features(vgg16_feats1, xyz_feats, reconstruction)
                    obj2_n_feats, obj2_reconstruction = self.legoformer.get_obj_features(vgg16_feats2, xyz_feats, reconstruction)
                else: 
                    obj1_n_feats, obj1_reconstruction = vgg16_feats1.squeeze(), None
                    obj2_n_feats, obj2_reconstruction = vgg16_feats2.squeeze(), None

                    # Correct for single-view. 
                    if len(obj1_n_feats.shape) == 2:
                        obj1_n_feats = obj1_n_feats.unsqueeze(1)
                        obj2_n_feats = obj2_n_feats.unsqueeze(1)

        elif self.cfg['train']['feats_backbone'] == 'pix2vox' or self.cfg['train']['feats_backbone'] == '3d-r2n2': 
            # Pre-extracted features
            obj1_n_feats, obj2_n_feats = obj_feats

        dtype = self.clip.visual.conv1.weight.dtype
        lang_feat = self.clip.token_embedding(lang_tokens.squeeze()).type(dtype)
        lang_feat = lang_feat + self.clip.positional_embedding.type(dtype)
        lang_feat = lang_feat.permute(1, 0, 2)
        lang_feat = self.clip.transformer(lang_feat)
        lang_feat = lang_feat.permute(1, 0, 2)
        lang_feat = self.clip.ln_final(lang_feat)

        # lang encoding with clip. # TODO Why doesn't CLIP mask zero-tokens? 
        if not self.cfg['transformer']['skip_clip']:
            
            # Aggregate CLIP langauge. 
            agg_lang_feat = lang_feat[torch.arange(lang_feat.shape[0]), lang_tokens.squeeze().argmax(dim=-1)] @ self.clip.text_projection

            """
            Separate stream v&l. 
            """
            vl1_feats = self.vl_mlp(torch.cat([agg_lang_feat, img1_feats], dim=-1))
            vl2_feats = self.vl_mlp(torch.cat([agg_lang_feat, img2_feats], dim=-1))
            """
            """

        """
        Transformer. 
        """
        if self.cfg['transformer']['head'] == 'transformer':
            
            # To cut compute time, clip tokens by maximal sentence length in batch. 
            max_length = (lang_tokens.squeeze() != 0).long().sum(dim=-1).max().item()
            lang_feat = lang_feat[:,:max_length]
            lang_tokens = lang_tokens.squeeze()[:,:max_length]

            lang_feat = lang_feat.float()

            # Project onto shared embedding space. 
            lang_enc = self.lang_fc(lang_feat)
            
            if self.cfg['data']['use_rgb_pc']:
                obj1_enc = obj1_n_feats
                obj2_enc = obj2_n_feats
            else: 
                obj1_enc = self.obj_fc(obj1_n_feats)
                obj2_enc = self.obj_fc(obj2_n_feats)
            

            # Perceiver pass for explicit voxels. 
            if self.use_peract:
                
                # Repeat latents to batch size. 
                latents = repeat(self.latents, 'n d -> b n d', b=obj1_enc.size(0))
                cross_attn, cross_ff = self.cross_attend_blocks
                
                # Project image features into transformer embedding space. 
                img1_feats = self.img_fc(img1_n_feats)
                img2_feats = self.img_fc(img2_n_feats)
            
                # Positional encoding and input prep. 
                obj1_in = torch.cat([obj1_enc, img1_feats, lang_enc], dim=1)
                obj2_in = torch.cat([obj2_enc, img2_feats, lang_enc], dim=1)
                
                obj1_in = self.pos_encoding[:,:obj1_in.size(1),:] + obj1_in
                obj2_in = self.pos_encoding[:,:obj2_in.size(1),:] + obj2_in
                
                # Perceiver pass per object. 
                obj1_latents = cross_attn(latents, context=obj1_in, mask=None) + latents
                obj2_latents = cross_attn(latents, context=obj2_in, mask=None) + latents
                
                obj1_latents = cross_ff(obj1_latents) + obj1_latents
                obj2_latents = cross_ff(obj2_latents) + obj2_latents
                
                for self_attn, self_ff in self.layers: 
                    obj1_latents = self_attn(obj1_latents) + obj1_latents
                    obj2_latents = self_attn(obj2_latents) + obj2_latents
                    
                    obj1_latents = self_ff(obj1_latents) + obj1_latents
                    obj2_latents = self_ff(obj2_latents) + obj2_latents
                
                # Decoder attention for CLS token. 
                cls = repeat(self.cls_token, '1 d -> b 1 d', b=obj1_latents.size(0))
                
                feats1 = self.decoder_cross_attn(cls, context=obj1_latents).squeeze()
                feats2 = self.decoder_cross_attn(cls, context=obj2_latents).squeeze()
                
                # Dummy return value. 
                lang_mask = None
                
            else: 

                # Concatenate tokens for transformer. 
                bz = lang_feat.size(0)
                cls_token = self.cls_token.unsqueeze(0).expand(bz, 1, -1)

                # Compute masks for transformer. 
                cls_mask = torch.full((bz, 1), False).to('cuda')
                lang_mask = (lang_tokens == 0.0).to('cuda')
                obj_mask = torch.full((bz, obj1_enc.size(1)), False).to('cuda')
                padding_mask = torch.cat([lang_mask, obj_mask, cls_mask], dim=1).to('cuda')

                # Pass tokens through transformer itself. 
                feats1 = torch.cat([lang_enc, obj1_enc, cls_token], dim=1)
                feats2 = torch.cat([lang_enc, obj2_enc, cls_token], dim=1)

                feats1 = self.transformer_pass(feats1, padding_mask, max_length)
                feats2 = self.transformer_pass(feats2, padding_mask, max_length)
            
            if not self.cfg['transformer']['skip_clip']:
                """
                Multi-stream fusion. 
                """
                feats1 = torch.cat([feats1, vl1_feats], dim=-1)
                feats2 = torch.cat([feats2, vl2_feats], dim=-1)
                """
                """

        else: 
        
            # TODO Deal with multiview case where we have to aggregate. 
            if len(obj1_n_feats.shape) == 3: 
                obj1_enc = torch.max(obj1_n_feats, dim=1)[0]
                obj2_enc = torch.max(obj2_n_feats, dim=1)[0]
            else: 
                obj1_enc = obj1_n_feats
                obj2_enc = obj2_n_feats

            # Project object embeddings. 
            obj1_enc = self.obj_fc(obj1_enc.squeeze())
            obj2_enc = self.obj_fc(obj2_enc.squeeze())

            """
            MLP with visiolinguistic stream.  
            """
            feats1 = self.mlp(torch.cat([vl1_feats, obj1_enc], dim=-1))
            feats2 = self.mlp(torch.cat([vl2_feats, obj2_enc], dim=-1))

            """
            """

            # Dummy return values. 
            obj1_reconstruction, obj2_reconstruction = (None, None)
            lang_mask = None

        # Score each object. 
        score1 = self.cls_fc(feats1)
        score2 = self.cls_fc(feats2)

        probs = torch.cat([score1, score2], dim=-1)

        # num steps taken (8 for all views)
        # TODO what does this do???
        bs = probs.shape[0]
        num_steps = torch.ones((bs)).to(dtype=torch.long, device=probs.device)
        num_steps = num_steps * self.num_views

        res = {
            'probs': probs,
            'is_visual': is_visual,
            'num_steps': num_steps,
            'reconstructions': (obj1_reconstruction, obj2_reconstruction),
            'gt_voxels': voxel_maps,
            'voxel_masks': voxel_masks, 
            'annotation': annotation,
            'lang_mask': lang_mask
        }

        if ans[0] > -1: 
            # one-hot labels of answers
            labels = F.one_hot(ans)
            res['labels'] = labels
        else: 
            res['labels'] = None

        return res

    # Additionally extracts object 
    def visualization_forward(self, batch):
        
        # Unpack features.  
        img_feats = batch['img_feats'] if 'img_feats' in batch else None        
        obj_feats = batch['obj_feats'] if 'obj_feats' in batch else None
        imgs = batch['images'] if 'images' in batch else None
        vgg16_feats = batch['vgg16_feats'] if 'vgg16_feats' in batch else None
        lang_tokens = batch['lang_tokens'].cuda()
        voxel_maps = batch['voxel_maps'] if 'voxel_maps' in batch else None
        voxel_masks= batch['voxel_masks'] if 'voxel_masks' in batch else None

        ans = batch['ans'].cuda() if 'ans' in batch else None
        (key1, key2) =  batch['keys']
        annotation = batch['annotation']
        is_visual = batch['is_visual']

        # TODO do we want to feed all of these into transformer, or just the aggregate? 
        # Load, aggregate, and process img features. 
        if img_feats: 
            img1_n_feats = img_feats[0].to(device=self.device).float()
            img2_n_feats = img_feats[1].to(device=self.device).float()  

            img1_feats = self.aggregator(img1_n_feats)
            img2_feats = self.aggregator(img2_n_feats)

            # Project into shared embedding space. 
            #img1_feats = self.img_fc(img1_feats)
            #img2_feats = self.img_fc(img2_feats)

        # Generate object features using legoformer.  
        # Right now we assume we've precomputed the VGG16 features and don't use raw images. 
        if self.cfg['train']['feats_backbone'] == 'legoformer':
            vgg16_feats1, vgg16_feats2 = vgg16_feats
            vgg16_feats1, vgg16_feats2 = vgg16_feats1.cuda(), vgg16_feats2.cuda()

            # Potentially skip legoformer all together and use VGG16 features directly. 
            if not self.cfg['transformer']['skip_legoformer']:
                
                # Also optionally get reconstruction output.
                reconstruction = self.cfg['data']['voxel_reconstruction']
                xyz_feats = self.cfg['transformer']['xyz_embeddings']
                obj1_n_feats, obj1_reconstruction = self.legoformer.get_obj_features(vgg16_feats1, xyz_feats, reconstruction)
                obj2_n_feats, obj2_reconstruction = self.legoformer.get_obj_features(vgg16_feats2, xyz_feats, reconstruction)
            else: 
                
                obj1_n_feats, obj1_reconstruction = vgg16_feats1.squeeze(), None
                obj2_n_feats, obj2_reconstruction = vgg16_feats2.squeeze(), None

                # Correct for single-view. 
                if len(obj1_n_feats.shape) == 2:
                    obj1_n_feats = obj1_n_feats.unsqueeze(1)
                    obj2_n_feats = obj2_n_feats.unsqueeze(1)

        elif self.cfg['train']['feats_backbone'] == 'pix2vox' or self.cfg['train']['feats_backbone'] == '3d-r2n2': 
            # Pre-extracted features
            obj1_n_feats, obj2_n_feats = obj_feats

        # lang encoding with clip. # TODO Why doesn't CLIP mask zero-tokens? 
        dtype = self.clip.visual.conv1.weight.dtype
        lang_feat = self.clip.token_embedding(lang_tokens.squeeze()).type(dtype)
        lang_feat = lang_feat + self.clip.positional_embedding.type(dtype)
        lang_feat = lang_feat.permute(1, 0, 2)
        lang_feat = self.clip.transformer(lang_feat)
        lang_feat = lang_feat.permute(1, 0, 2)
        lang_feat = self.clip.ln_final(lang_feat)
 
        # Aggregate CLIP langauge. 
        agg_lang_feat = lang_feat[torch.arange(lang_feat.shape[0]), lang_tokens.squeeze().argmax(dim=-1)] @ self.clip.text_projection

        """
        Transformer. 
        """
        if self.cfg['transformer']['head'] == 'transformer':
            # To cut compute time, clip tokens by maximal sentence length in batch. 
            max_length = (lang_tokens.squeeze() != 0).long().sum(dim=-1).max().item()
            lang_feat = lang_feat[:,:max_length]
            lang_tokens = lang_tokens.squeeze()[:,:max_length]

            lang_feat = lang_feat.float()

            # Project onto shared embedding space. 
            lang_enc = self.lang_fc(lang_feat)
            obj1_enc = self.obj_fc(obj1_n_feats)
            obj2_enc = self.obj_fc(obj2_n_feats)

            # Concatenate tokens for transformer. 
            bz = lang_feat.size(0)
            cls_token = self.cls_token.unsqueeze(0).expand(bz, 1, -1)

            # Compute masks for transformer. 
            cls_mask = torch.full((bz, 1), False).to('cuda')
            lang_mask = (lang_tokens == 0.0).to('cuda')
            obj_mask = torch.full((bz, obj1_enc.size(1)), False).to('cuda')
            padding_mask = torch.cat([lang_mask, obj_mask, cls_mask], dim=1).to('cuda')

            # Pass tokens through transformer itself. 
            feats1 = torch.cat([lang_enc, obj1_enc, cls_token], dim=1)
            feats2 = torch.cat([lang_enc, obj2_enc, cls_token], dim=1)

            feats1, attn_weights1 = self.transformer_pass(feats1, padding_mask, max_length, get_weights=True)
            feats2, attn_weights2 = self.transformer_pass(feats2, padding_mask, max_length, get_weights=True)

            """
            Separate stream v&l. 
            """
            vl1_feats = self.vl_mlp(torch.cat([agg_lang_feat, img1_feats], dim=-1))
            vl2_feats = self.vl_mlp(torch.cat([agg_lang_feat, img2_feats], dim=-1))
            """
            """
             
            """
            Multi-stream fusion. 
            """
            score1 = self.cls_fc(torch.cat([feats1, vl1_feats], dim=-1))
            score2 = self.cls_fc(torch.cat([feats2, vl2_feats], dim=-1))
            """
            """
     
        else: 
        
            # TODO Deal with multiview case where we have to aggregate. 
            if len(obj1_n_feats) == 3: 
                obj1_enc = torch.max(obj1_n_feats, dim=1)[0]
                obj2_enc = torch.max(obj2_n_feats, dim=1)[0]
            else: 
                obj1_enc = obj1_n_feats
                obj2_enc = obj2_n_feats

            """
            MLP 
            """
            feats1 = torch.cat([img1_feats, lang_enc.squeeze(), obj1_enc], dim=-1)
            feats2 = torch.cat([img2_feats, lang_enc.squeeze(), obj2_enc], dim=-1)

            score1 = self.mlp(feats1)
            score2 = self.mlp(feats2)
            """
            """
        
        # Score each object. 
        probs = torch.cat([score1, score2], dim=-1)

        # num steps taken (8 for all views)
        # TODO what does this do???
        bs = lang_enc.shape[0]
        num_steps = torch.ones((bs)).to(dtype=torch.long, device=lang_enc.device)
        num_steps = num_steps * self.num_views

        res = {
            'probs': probs,
            'is_visual': is_visual,
            'num_steps': num_steps,
            'reconstructions': (obj1_reconstruction, obj2_reconstruction),
            'gt_voxels': voxel_maps,
            'voxel_masks': voxel_masks, 
            'attn_maps': (attn_weights1, attn_weights2),
            'annotation': annotation,
            'lang_mask': lang_mask
        }

        if not ans.sum() > 0: 
            # one-hot labels of answers
            labels = F.one_hot(ans)
            res['labels'] = labels

        return res


    def training_step(self, batch, batch_idx):
        out = self.forward(batch)

        # classifier loss
        losses = self._criterion(out)

        # Will contain all logging for wandb.
        for loss in losses.keys(): 
            self.log_dict['tr/{}'.format(loss)] = losses[loss]

        # Compute correct.  
        correct = self.check_correct(out['labels'], out['probs'])

         # Additionally evaluate model reconstruction performance. 
        if self.cfg['data']['voxel_reconstruction']: 

            # Unpack volumes.  
            pred_voxel1, pred_voxel2 = out['reconstructions']
            gt_voxel1, gt_voxel2 = out['gt_voxels']
            vmask1, vmask2 = out['voxel_masks']

            # Binarize for evaluation. 
            pred_voxel1, pred_voxel2 = pred_voxel1.__ge__(0.3), pred_voxel2.__ge__(0.3)
            gt_voxel1, gt_voxel2 = gt_voxel1.__ge__(0.5), gt_voxel2.__ge__(0.5)
            vmask = torch.cat(out['voxel_masks'])

            # Compute F-score and IoU for volumes. 
            iou1 = calculate_iou(pred_voxel1, gt_voxel1, compute_mean=False)
            iou2 = calculate_iou(pred_voxel2, gt_voxel2, compute_mean=False)
            iou = torch.cat([iou1, iou2])
    
            # Compute average IoU
            self.log_dict['tr/iou'] = (iou * vmask).sum() / vmask.sum()
            # TODO need to mask out invalid gt voxels! 

            #fs1 = calculate_fscore(pred_voxel1, gt_voxel1)
            #fs2 = calculate_fscore(pred_voxel2, gt_voxel2)

        self.log_dict['tr/acc'] = (correct.sum() / correct.size(0)).detach().cpu().numpy()

        # Compute visualization of correctness for some samples for debugging model. 
        """
        if self.step_num % self.cfg['wandb']['logger']['img_log_freq'] == 0: 
            self.visualize_examples(batch, out, 20, 'train')
        """
        
        return dict(
            loss=losses['loss']
        )

    def visualize_examples(self, batch, out, n_examples, name): 
        
        # Compute correct.  
        probs = out['probs']
        labels = out['labels']

        # Load images. 
        keys1, keys2 = batch['keys']
        keys1, keys2 = keys1[:n_examples], keys2[:n_examples]

        # Compute which examples are correct. 
        all_correct = self.check_correct(labels, probs)
        guesses = probs.argmax(dim=1)

        for idx in range(n_examples): 

            # Check if correct, record guess, annotation, and images. 
            correct = all_correct[idx]
            guess = guesses[idx]
            annotation = batch['annotation'][idx]
            visual = batch['is_visual'][idx]

            # Load images. 
            start_idx = 14 - self.num_views
            key1, key2 = keys1[idx], keys2[idx]
            imgs1, imgs2 = [], []

            # Path prefixes to images. 
            img_dir = os.path.join(self.cfg['root_dir'], 'data/screenshots')
            dir1 = os.path.join(img_dir, key1)
            dir2 = os.path.join(img_dir, key2)
        
            for i in range(self.num_views): 

                # Get absolute paths to images. 
                img_idx = start_idx + i
                img1_path = os.path.join(dir1, '{}-{}.png'.format(key1, img_idx))
                img2_path = os.path.join(dir2, '{}-{}.png'.format(key2, img_idx))

                # Load images themselves. 
                img1 = cv2.imread(img1_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
                img2 = cv2.imread(img2_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0

                imgs1.append(torch.from_numpy(img1).permute(2, 0, 1))
                imgs2.append(torch.from_numpy(img2).permute(2, 0, 1))

            # Make single image grid for visualization. 
            all_imgs = imgs1 + imgs2 
            viz_img= make_grid(all_imgs, self.num_views).permute(1, 2, 0).numpy()

            # Caption will contain annotation and pertinent information. 
            caption = '{}\nCorrect: {}\nGuess: {}\nProbs:{}\nVisual: {}'.format(annotation, correct, guess, probs[idx], visual)
            self.log_dict['{}/ex-{}'.format(name, idx)] = wandb.Image(viz_img, caption=caption)

    def on_after_backward(self):

        if self.log_data:

            # Log weights and parameters every n training steps. 
            if self.step_num % self.cfg['wandb']['logger']['param_log_freq'] == 0: 
                for name, param in self.named_parameters(): 
                    if not param.grad is None:  

                        # Weights
                        weights = param.data.view(-1).detach().cpu().numpy()
                        self.log_dict['weights/mean/{}'.format(name)] = weights.mean()
                        self.log_dict['weights/abs_mean/{}'.format(name)] = np.abs(weights).mean()
                        self.log_dict['weights/std/{}'.format(name)] = weights.std()
                        self.log_dict['weights/min/{}'.format(name)] = weights.min()
                        self.log_dict['weights/max/{}'.format(name)] = weights.max()

                        # Grad
                        grad = param.grad.view(-1).detach().cpu().numpy()
                        self.log_dict['grad/mean/{}'.format(name)] = grad.mean()
                        self.log_dict['grad/abs_mean/{}'.format(name)] = np.abs(grad).mean()
                        self.log_dict['grad/std/{}'.format(name)] = grad.std()
                        self.log_dict['grad/min/{}'.format(name)] = grad.min()
                        self.log_dict['grad/max/{}'.format(name)] = grad.max()

                        # Add weights. 
                        self.log_dict['weights/values/{}'.format(name)] = \
                                wandb.Histogram(weights)

                        # Add gradients. 
                        self.log_dict['grads/values/{}'.format(name)] = \
                                wandb.Histogram(grad)

    def on_train_batch_end(self, trainer, pl_module, batch, batch_idx): 
        
        # Update logs. 
        if self.step_num % self.cfg['wandb']['logger']['acc_log_freq'] == 0: 
            pass#wandb.log(self.log_dict)
        
        self.step_num += 1
        log_dict = {'step_num': self.step_num}

    def check_correct(self, labels, probs):
        guess = probs.argmax(dim=1)
        labels = labels.argmax(dim=1)
        correct = torch.eq(labels, guess).float()
        
        return correct

    def validation_step(self, batch, batch_idx):
        out = self.forward(batch)
        losses = self._criterion(out)

        probs = out['probs']
        labels = out['labels']
        visual = out['is_visual']
        num_steps = out['num_steps']

        # Keep track of predictions. 
        self.val_predictions['labels'].append(labels.cpu().numpy())
        self.val_predictions['probs'].append(probs.cpu().numpy())
        self.val_predictions['visual'].append(visual.long().cpu().numpy())

        probs = F.softmax(probs, dim=-1)
        metrics = self.compute_metrics(labels, losses, probs, visual, num_steps, out)
        
        # Increment validation steps. 
        self.val_step_num += 1

        # Visualize results for first batch from validation set. 
        """
        if self.val_step_num == 1 and self.epoch_num % self.cfg['wandb']['logger']['val_img_log_epoch_freq'] == 0: 
            self.visualize_examples(batch, out, 20, 'val')
        """

        self.val_step_num += 1

        return dict(
            val_loss=metrics['val_loss'],
            val_acc=metrics['val_acc'],
            metrics=metrics
        )

    def compute_metrics(self, labels, losses, probs, visual, num_steps, out):
        val_total = probs.shape[0]
        
        pred_voxels = out['reconstructions'] if 'reconstructions' in out else None
        gt_voxels = out['gt_voxels'] if 'gt_voxels' in out else None
        vmasks = out['voxel_masks'] if 'voxel_masks' in out else None
        
        # TODO change naming scheme to accomodate for test set as well. 

        # Compute correct by index in batch. 
        correct = self.check_correct(labels, probs)
        val_correct = correct.sum().item()

        # See which visual examples are correct and which aren't. 
        visual_total = visual.float().sum().item()
        visual_correct = (visual.view(-1).float() * correct).sum().item()

        nonvis_total = float(val_total) - visual_total
        nonvis_correct = val_correct - visual_correct

        val_acc = float(val_correct) / val_total
        
        # Take care in cases where we're only using one of the splits. 
        if visual_total > 0: 
            val_visual_acc = float(visual_correct) / visual_total
        else: 
            val_visual_acc = 0.0
        
        if nonvis_total > 0: 
            val_nonvis_acc = float(nonvis_correct) / nonvis_total
        else:
            val_nonvis_acc = 0.0

        return_dict = dict(
            val_acc=val_acc,
            val_correct=val_correct,
            val_total=val_total,
            val_visual_acc=val_visual_acc,
            val_visual_correct=visual_correct,
            val_visual_total=visual_total,
            val_nonvis_acc=val_nonvis_acc,
            val_nonvis_correct=nonvis_correct,
            val_nonvis_total=nonvis_total
        )

        for loss in losses.keys(): 
            return_dict['val_{}'.format(loss)] = losses[loss]

        # Additionally evaluate model reconstruction performance. 
        if self.cfg['data']['voxel_reconstruction']: 

            assert (pred_voxels != None) and (gt_voxels != None)

            # Unpack volumes.  
            pred_voxel1, pred_voxel2 = pred_voxels
            gt_voxel1, gt_voxel2 = gt_voxels

            # Binarize for evaluation. 
            pred_voxel1, pred_voxel2 = pred_voxel1.__ge__(0.3), pred_voxel2.__ge__(0.3)
            gt_voxel1, gt_voxel2 = gt_voxel1.__ge__(0.5), gt_voxel2.__ge__(0.5)
            vmask = torch.cat(vmasks)

            # Compute F-score and IoU for volumes. 
            iou1 = calculate_iou(pred_voxel1, gt_voxel1, compute_mean=False)
            iou2 = calculate_iou(pred_voxel2, gt_voxel2, compute_mean=False)
            iou = torch.cat([iou1, iou2])

            # Sum to aggregate with minimal memory usage, will take mean at end of epoch. 
            return_dict['iou'] = (iou * vmask).sum() / vmask.sum()
            # TODO need to mask out invalid gt voxels! 

            #fs1 = calculate_fscore(pred_voxel1, gt_voxel1)
            #fs2 = calculate_fscore(pred_voxel2, gt_voxel2)

        return return_dict

    def training_epoch_end(self, output): 
        self.epoch_num += 1

    def on_validation_start(self): 
        self.val_step_num = 0

    def validation_epoch_end(self, all_outputs, mode='vl'):
        sanity_check = True

        # Consolidate all predictions. 
        """
        self.val_predictions['probs'] = np.concatenate(self.val_predictions['probs'], axis=0)
        self.val_predictions['labels'] = np.concatenate(self.val_predictions['labels'], axis=0)
        self.val_predictions['visual'] = np.concatenate(self.val_predictions['visual'], axis=0)
        """

        res = {
            'val_loss': 0.0,

            'val_correct': 0,
            'val_total': 0,

            'val_visual_correct': 0,
            'val_visual_total': 0,

            'val_nonvis_correct': 0,
            'val_nonvis_total': 0,

            'val_iou': 0
        }

        for output in all_outputs:
            metrics = output['metrics']
            res['val_loss'] += metrics['val_loss'].item()
            res['val_correct'] += metrics['val_correct']
            res['val_total'] += metrics['val_total']
            
            if res['val_total'] > 128:
                sanity_check = False

            res['val_visual_correct'] += metrics['val_visual_correct']
            res['val_visual_total'] += metrics['val_visual_total']

            res['val_nonvis_correct'] += metrics['val_nonvis_correct']
            res['val_nonvis_total'] += metrics['val_nonvis_total']

            if 'iou' in metrics: 
                res['val_iou'] += metrics['iou']

        res['val_loss'] = float(res['val_loss']) / len(all_outputs)
        res['val_acc'] = float(res['val_correct']) / res['val_total']
        
        if res['val_visual_total'] > 0: 
            res['val_visual_acc'] = float(res['val_visual_correct']) / res['val_visual_total']
        else: 
            res['val_visual_acc'] = 0.0
        
        if res['val_nonvis_total'] > 0: 
            res['val_nonvis_acc'] = float(res['val_nonvis_correct']) / res['val_nonvis_total']
        else: 
            res['val_nonvis_acc'] = 0.0

        # Compute IoU metric. # TODO correct for masking out invalid voxelmaps. (333/7881 in dataset).  
        res['val_iou'] = float(res['val_iou']) / (res['val_total'] * 2) # Have two objects per example.

        res = {
            f'{mode}/loss': res['val_loss'],
            f'{mode}/acc': res['val_acc'],
            f'{mode}/acc_visual': res['val_visual_acc'],
            f'{mode}/acc_nonvis': res['val_nonvis_acc'],
            f'{mode}/iou': res['val_iou']
        }

        if not sanity_check:  # only check best conditions and dump data if this isn't a sanity check

            # test (ran once at the end of training)
            if mode == 'test':
                self.best_test_res = dict(res)

            # val (keep track of best results)
            else:
                if res[f'{mode}/acc'] > self.best_val_acc:
                    self.best_val_acc = res[f'{mode}/acc']
                    self.best_val_res = dict(res)
                    
                    # Store predictions. 
                    """
                    preds_path = os.path.join(self.save_path, 'val_preds.npz')
                    np.savez(
                        preds_path, 
                        probs=self.val_predictions['probs'],
                        labels=self.val_predictions['labels'],
                        visual=self.val_predictions['visual']
                    )
                    """

            # results to save
            results_dict = self.best_test_res if mode == 'test' else self.best_val_res

            best_loss = results_dict[f'{mode}/loss']
            best_acc = results_dict[f'{mode}/acc']
            best_acc_visual = results_dict[f'{mode}/acc_visual']
            best_acc_nonvis = results_dict[f'{mode}/acc_nonvis']

            seed = self.cfg['train']['random_seed']
            json_file = os.path.join(self.save_path, f'{mode}-results-{seed}.json')

            # save results
            with open(json_file, 'w') as f:
                json.dump(results_dict, f, sort_keys=True, indent=4)

            # print best result
            print("\nBest-----:")
            print(f'Best {mode} Acc: {best_acc:0.5f} | Visual {best_acc_visual:0.5f} | Nonvis: {best_acc_nonvis:0.5f} | Val Loss: {best_loss:0.8f} ')
            print("------------")

        # Add results to log dictionary. 
        pass#wandb.log(res, self.step_num)

        # Re-initialize val predictions buffer. 
        self.val_predictions = {
            'probs': [],
            'labels': [],
            'visual': []
        }

        return dict(
            val_loss=res[f'{mode}/loss'],
            val_acc=res[f'{mode}/acc'],
            val_visual_acc=res[f'{mode}/acc_visual'],
            val_nonvis_acc=res[f'{mode}/acc_nonvis'],

        )

    def test_step(self, batch, batch_idx):
        all_view_results = {}
        
        out = self.forward(batch)
         
        probs = out['probs']
        num_steps = out['num_steps']
        objects = batch['keys']
        annotation = batch['annotation']
        labels = out['labels'] if 'labels' in out else None
        visual = out['is_visual']
        probs = F.softmax(probs, dim=-1)
        pred_ans = probs.argmax(-1)

        if type(labels) != type(None): 
            losses = self._criterion(out)
            metrics = self.compute_metrics(labels, losses, probs, visual, num_steps, out)

        for view in range(self.num_views): 
            all_view_results[view] = dict(
                annotation=annotation,
                objects=objects, 
                pred_ans=pred_ans,
                num_steps = num_steps
            )

        res = dict(
            all_view_results = all_view_results
        )

        if type(labels) != type(None): 
            res['metrics'] = metrics

        return res

    def test_epoch_end(self, all_outputs, mode='test'):
        res = {
            'val_loss': 0.0,

            'val_correct': 0,
            'val_total': 0,

            'val_visual_correct': 0,
            'val_visual_total': 0,

            'val_nonvis_correct': 0,
            'val_nonvis_total': 0,

            'val_iou': 0.0
        }


        if 'metrics' in all_outputs[0]: 
            for output in all_outputs:
                metrics = output['metrics']
                res['val_loss'] += metrics['val_loss'].item()
                res['val_correct'] += metrics['val_correct']
                res['val_total'] += metrics['val_total']
                
                if res['val_total'] > 128:
                    sanity_check = False

                res['val_visual_correct'] += metrics['val_visual_correct']
                res['val_visual_total'] += metrics['val_visual_total']

                res['val_nonvis_correct'] += metrics['val_nonvis_correct']
                res['val_nonvis_total'] += metrics['val_nonvis_total']

                if 'iou' in metrics: 
                    res['val_iou'] += metrics['iou']

            res['val_loss'] = float(res['val_loss']) / len(all_outputs)
            res['val_acc'] = float(res['val_correct']) / res['val_total']
            res['val_visual_acc'] = float(res['val_visual_correct']) / res['val_visual_total']
            res['val_nonvis_acc'] = float(res['val_nonvis_correct']) / res['val_nonvis_total']

            # IoU computation. 
            res['val_iou'] = float(res['val_iou']) / (res['val_total'] * 2)

            res = {
                f'{mode}/loss': res['val_loss'],
                f'{mode}/acc': res['val_acc'],
                f'{mode}/acc_visual': res['val_visual_acc'],
                f'{mode}/acc_nonvis': res['val_nonvis_acc'],
                f'{mode}/iou': res['val_iou']
            }

            print('{mode} results:')
            for key, val in res.items(): 
                print('{}: {}'.format(key, val))

        # Actually compute test results to get results file.  
        test_results = {v: list() for v in range(self.num_views)}

        for out in all_outputs:
            for view in range(self.num_views): 
                view_res = out['all_view_results']
                bs = view_res[view]['pred_ans'].shape[0]

                for b in range(bs):
                    test_results[view].append({
                        'annotation': view_res[view]['annotation'][b],
                        'objects': (
                            view_res[view]['objects'][0][b],
                            view_res[view]['objects'][1][b],
                        ),
                        'pred_ans': int(view_res[view]['pred_ans'][b]),
                        'num_steps': int(view_res[view]['num_steps'][b]),
                    })

        if mode == 'test': 
            test_pred_save_path = self.save_path
            if not os.path.exists(test_pred_save_path):
                os.makedirs(test_pred_save_path)

            model_type = self.__class__.__name__.lower()
            json_file = os.path.join(test_pred_save_path, f'{model_type}_test_results.json')
            
            print('Saving results to: {}'.format(json_file))
            
            with open(json_file, 'w') as f:

                json.dump(test_results, f, sort_keys=True, indent=4)
