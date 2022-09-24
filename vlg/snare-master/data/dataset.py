import os
import json
import torch
import torch.utils.data
import clip
import cv2
import torchvision.models
import h5py
from PIL import Image

import numpy as np
import gzip
import json
import pdb
import tqdm
from einops import rearrange
import pickle
from tqdm import tqdm
from pyhocon import ConfigFactory
import imageio

import legoformer.data as transforms
from legoformer.data.dataset import ShapeNetDataset
from data.verify_shapenet import get_snare_objs
from pixelnerf.src.model import make_model
from pixelnerf.src.util.util import gen_rays, pose_spherical
from pixelnerf.src.render.nerf import NeRFRenderer
from pixelnerf.src.util import repeat_interleave
from dotmap import DotMap

class CLIPGraspingDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, mode='train', legoformer_data_module=None, img_feat_file=None, obj2idx_mapping=None):
        self.total_views = 14
        self.cfg = cfg
        self.mode = mode
        self.folds = os.path.join(self.cfg['data']['amt_data'], self.cfg['data']['folds'])
        self.feats_backbone = self.cfg['train']['feats_backbone']

        self.n_views = self.cfg['data']['n_views']

        print("Num views: {}".format(self.n_views))

        self.load_entries()
        self.load_extracted_features()

        # Paths to ShapeNet rendered images, whether original or custom rendered. 
        if cfg['data']['custom_renders']: 
            self.img_path = cfg['data']['custom_render_path']
        else:    
            self.img_path = os.path.join(self.cfg['root_dir'], 'data/screenshots')

        # Paths to shapenet objects. # TODO Does this code still generalize to LegoFormer model needs? 
        self.shapenet_path = os.path.join(self.cfg['root_dir'], 'data/screenshots') 

        # Get transforms for preprocessing ShapeNet images. 
        if legoformer_data_module: 
            self.transforms = legoformer_data_module.get_eval_transforms(legoformer_data_module.cfg_data.transforms)
        else: 
            self.transforms = None

        # Keep camera parameters if they are given. (Used for PixelNeRF)
        if self.cfg['train']['model'] == 'pixelnerf': 
            
            # Load or compute pixelnerf features. 
            self.load_pixelnerf_data(cfg['pixelnerf']['camera_param_path'])

        # Use images during loading or not (if feeding straight into LegoFormer). 
        if self.cfg['train']['model'] == 'transformer': # TODO generalize to pixelnerf.  
            self.use_imgs = True
        else: 
            self.use_imgs = False

    def load_pixelnerf_data(self, camera_params):

        # Paths for precomputed pixelnerf features. 
        feat_type = self.cfg['pixelnerf']['feat_type']

        if feat_type == 'pre-query':
            self.pixelnerf_feat_dir = self.cfg['pixelnerf']['feature_dir']

        elif feat_type == 'coarse':
            self.pixelnerf_feat_dir = self.cfg['pixelnerf']['coarse_feature_dir']
        
        elif feat_type == 'fine':
            self.pixelnerf_feat_dir = self.cfg['pixelnerf']['fine_feature_dir']

        # Load camera parameters for pixelnerf. 
        self.load_cam_poses()

        # Precompute and save to disk if needed. 
        if not (os.path.isdir(self.pixelnerf_feat_dir)):
            os.mkdir(self.pixelnerf_feat_dir)
            self.compute_pixelnerf_features(self.pixelnerf_feat_dir, feat_type)

    def load_cam_poses(self):
        # Load camera parameters. 
        camera_params = self.cfg['pixelnerf']['camera_param_path']     
        self.camera_params = np.load(camera_params, allow_pickle=True).item()
        
        self._coord_trans_world = torch.tensor(
            [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
            dtype=torch.float32,
        )
        self._coord_trans_cam = torch.tensor(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
            dtype=torch.float32,
        )

        # Pre-compute camera matrices since these won't change. 
        self.cam_poses = []

        for i in range(8):
        
            # Load inverse of world matrix. 
            wmi = self.camera_params['world_mat_inv_{}'.format(i)]

            # Compute pose. 
            pose = (
                self._coord_trans_world
                @ torch.tensor(wmi, dtype=torch.float32)
                @ self._coord_trans_cam
            )

            self.cam_poses.append(pose.numpy())

        self.cam_poses = np.asarray(self.cam_poses)

    def compute_pixelnerf_features(self, feature_dir, feat_type):

        # Get the objects we need features for. 
        snare_objs = list(get_snare_objs())

        # Load pre-trained pixelnerf for feature extraction. 
        pn_cfg = ConfigFactory.parse_file(self.cfg['pixelnerf']['pn_cfg'])
        pixelnerf = make_model(pn_cfg["model"]).cuda()
        pixelnerf.eval()

        pn_state_dict = torch.load(self.cfg['pixelnerf']['pn_checkpoint'], map_location='cuda:0')
        pixelnerf.load_state_dict(pn_state_dict, strict=True)

        # Compute features. 
        print('Pre-computing pixelnerf features...')

        # Camera parameters. 
        focal = torch.tensor((119.4256,)).cuda() # 119.43 3.7321
        cam_poses = torch.tensor(self.cam_poses).unsqueeze(0).cuda()
        c = None

        for idx in tqdm(range(len(snare_objs))): 
            
            # Get object key. 
            obj = snare_objs[idx]

            """
            with open('/home/rcorona/data/NMR_Dataset/03001627/test.lst', 'r') as f: 
                lines = [l.strip() for l in f]

                pdb.set_trace()

                for i in range(len(lines)):
                    if lines[i] in snare_objs: 
                        print(lines[i])
                        print(i)
                        exit()
            """

            # Load image. 
            imgs = torch.tensor(self.get_imgs(obj)).unsqueeze(0).cuda()

            # Encode image.
            with torch.no_grad():
                pixelnerf.encode(imgs, cam_poses, focal, c=c)

                # If just using first encoder, then we're done. 
                if feat_type == 'pre-query':
                    feat = pixelnerf.encoder.latent

                else: 

                    ray_bz = 50000
                    renderer = NeRFRenderer.from_conf(
                        pn_cfg["renderer"], lindisp=False, eval_batch_size=ray_bz,
                    ).cuda()

                    if feat_type == 'coarse':
                        feat = self.pixelnerf_coarse_features(pixelnerf, renderer, focal, c)

                    elif feat_type == 'fine':
                        pass # TODO 

            #pixelnerf_features[idx] = latent.cpu().numpy()
            # Store features. 
            feat_path = os.path.join(feature_dir, '{}.npy'.format(obj))
            np.save(feat_path, feat.cpu().numpy())

    def composite(self, z_samp, rays, sb, eval_batch_size, pixelnerf,
                    num_views_per_obj, coarse=True):

        B, K = z_samp.shape

        deltas = z_samp[:, 1:] - z_samp[:, :-1]  # (B, K-1)
        delta_inf = rays[:, -1:] - z_samp[:, -1:]
        deltas = torch.cat([deltas, delta_inf], -1)  # (B, K)

        # (B, K, 3)
        points = rays[:, None, :3] + z_samp.unsqueeze(2) * rays[:, None, 3:6]
        points = points.reshape(-1, 3)  # (B*K, 3)

        points = points.reshape(
            sb, -1, 3
        )  # (SB, B'*K, 3) B' is real ray batch size
        eval_batch_dim = 1

        split_points = torch.split(points, eval_batch_size, dim=eval_batch_dim)
        dim1 = K
        viewdirs = rays[:, None, 3:6].expand(-1, dim1, -1)  # (B, K, 3)

        viewdirs = viewdirs.reshape(sb, -1, 3)  # (SB, B'*K, 3)
        split_viewdirs = torch.split(
            viewdirs, eval_batch_size, dim=eval_batch_dim
        )

        val_all = []

        for xyz, viewdirs in zip(split_points, split_viewdirs):

            SB, B, _ = xyz.shape
            NS = 8
            poses = pixelnerf.poses

            # Transform query points into the camera spaces of the input views
            xyz = repeat_interleave(xyz, NS)  # (SB*NS, B, 3)
            xyz_rot = torch.matmul(poses[:, None, :3, :3], xyz.unsqueeze(-1))[
                ..., 0
            ]

            xyz = xyz_rot + poses[:, None, :3, 3]
            z_feature = xyz_rot.reshape(-1, 3)
            z_feature = pixelnerf.code(z_feature)

            viewdirs = viewdirs.reshape(SB, B, 3, 1)
            viewdirs = repeat_interleave(viewdirs, NS)  # (SB*NS, B, 3, 1)
            viewdirs = torch.matmul(
                poses[:, None, :3, :3], viewdirs
            )  # (SB*NS, B, 3, 1)
            viewdirs = viewdirs.reshape(-1, 3)  # (SB*B, 3)
            z_feature = torch.cat(
                (z_feature, viewdirs), dim=1
            )  # (SB*B, 4 or 6)

            mlp_input = z_feature

            pdb.set_trace()

            # Grab encoder's latent code.
            uv = -xyz[:, :, :2] / xyz[:, :, 2:]  # (SB, B, 2)
            uv *= repeat_interleave(
                pixelnerf.focal.unsqueeze(1), NS if pixelnerf.focal.shape[0] > 1 else 1
            )

            uv += repeat_interleave(
                pixelnerf.c.unsqueeze(1), NS if pixelnerf.c.shape[0] > 1 else 1
            )  # (SB*NS, B, 2)
            latent = pixelnerf.encoder.index(
                uv, None, pixelnerf.image_shape
            )  # (SB * NS, latent, B)

            latent = latent.transpose(1, 2).reshape(
                -1, pixelnerf.latent_size
            )  # (SB * NS * B, latent)

            pdb.set_trace()

            mlp_input = torch.cat((latent, z_feature), dim=-1)

            # Camera frustum culling stuff, currently disabled
            combine_index = None
            dim_size = None

            # Run main NeRF network
            if coarse: 

                mlp_output, feat = pixelnerf.mlp_coarse(
                    mlp_input,
                    combine_inner_dims=(num_views_per_obj, B),
                    combine_index=combine_index,
                    dim_size=dim_size,
                    return_feat=True,
                )

                pdb.set_trace()
            else: 
                mlp_output, feat = pixelnerf.mlp_fine(
                    mlp_input,
                    combine_inner_dims=(num_views_per_obj, B),
                    combine_index=combine_index,
                    dim_size=dim_size,
                    return_feat=True
                )

            rgb = mlp_output[..., :3]
            sigma = mlp_output[..., 3:4]

            output_list = [torch.sigmoid(rgb), torch.relu(sigma)]
            output = torch.cat(output_list, dim=-1)
            output = output.reshape(SB, B, -1)

            val_all.append(output)

        points = None
        viewdirs = None
        # (B*K, 4) OR (SB, B'*K, 4)

        out = torch.cat(val_all, dim=eval_batch_dim)
        out = out.reshape(B, K, -1)  # (B, K, 4 or 5)

        rgbs = out[..., :3]  # (B, K, 3)
        sigmas = out[..., 3]  # (B, K)

        alphas = 1 - torch.exp(-deltas * torch.relu(sigmas))  # (B, K)
        deltas = None
        sigmas = None
        alphas_shifted = torch.cat(
            [torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10], -1
        )  # (B, K+1) = [1, a1, a2, ...]
        T = torch.cumprod(alphas_shifted, -1)  # (B)
        weights = alphas * T[:, :-1]  # (B, K)
        alphas = None
        alphas_shifted = None

        rgb_final = torch.sum(weights.unsqueeze(-1) * rgbs, -2)  # (B, 3)
        depth_final = torch.sum(weights * z_samp, -1)  # (B)
        
        # White background
        pix_alpha = weights.sum(dim=1)  # (B), pixel alpha
        rgb_final = rgb_final + 1 - pix_alpha.unsqueeze(-1)  # (B, 3)

        composite = (
            weights,
            rgb_final,
            depth_final,
        )

        return composite

    def pixelnerf_coarse_features(self, pixelnerf, renderer, focal, c):

        # TODO Do these need to change for our renderings???
        # Parameters for generating query poses. 
        z_near = 1.2
        z_far = 4.0
        num_views = 40
        W = 64
        H = 64 
        elevation = -30.0
        radius = (z_near + z_far) * 0.5
        scale = 1.0
        sb = 1
        num_views_per_obj = 8
        ray_batch_size = 50000
        superbatch_size = 1

        # NeRF Renderer. 

        # Sample poses in 360 degrees.  
        render_poses = torch.stack(
            [
                pose_spherical(angle, elevation, radius)
                for angle in np.linspace(-180, 180, num_views + 1)[:-1]
            ],
            0,
        )  # (NV, 4, 4)

        render_rays = gen_rays(
            render_poses,
            W,
            H,
            focal * scale,
            z_near,
            z_far,
            c=c * scale if c is not None else None,
        ).cuda().view(-1, 8)

        all_rgb = []

        for rays in tqdm(torch.split(render_rays.view(-1, 8), ray_batch_size, dim=0)):

            z_coarse = renderer.sample_coarse(rays)

            # Coarse image features. 
            coarse_composite = self.composite(
                z_coarse, 
                rays, 
                sb, 
                rays.size(0), 
                pixelnerf, 
                num_views_per_obj, 
                True
            )

            outputs = renderer._format_outputs(coarse_composite, superbatch_size, False,)
            outputs = DotMap(outputs,)

            # Fine image features. 
            all_samps = [z_coarse]
            all_samps.append(renderer.sample_fine(rays, coarse_composite[0].detach()))
            all_samps.append(renderer.sample_fine_depth(rays, coarse_composite[2]))

            z_combine = torch.cat(all_samps, dim=-1)  # (B, Kc + Kf)
            z_combine_sorted, argsort = torch.sort(z_combine, dim=-1)

            fine_composite = self.composite(
                z_combine_sorted, 
                rays, 
                sb, 
                rays.size(0), 
                pixelnerf, 
                num_views_per_obj, 
                False
            )

            outputs.fine = renderer._format_outputs(
                fine_composite, superbatch_size, want_weights=False,
            )

            rgb, depth = outputs.fine.rgb, outputs.fine.depth
            all_rgb.append(rgb[0])

        frames = torch.cat(all_rgb).view(-1, 64, 64, 3)

        # Write rendered video.
        vid_path = '/home/rcorona/2022/lang_nerf/vlg/snare-master/test.mp4' 

        imageio.mimwrite(
        vid_path, (frames.cpu().numpy() * 255).astype(np.uint8), fps=30, quality=8
        )

        print('Wrote to: {}'.format(vid_path))

        exit()

    def preprocess_obj_feats(self): 

        # Don't need to pre-extract legoformer features. 
        if self.feats_backbone == 'legoformer':
            return 

        # Chose model. 
        if self.feats_backbone == 'pix2vox': 
            model = Pix2Vox(self.cfg)
        elif self.feats_backbone == '3d-r2n2': 
            raise NotImplementedError

        # First make set of all objects that don't yet have a feature matrix. 
        missing_objs = set()
        done_objs = set()

        for obj in os.listdir(self.shapenet_path): 
            
            # Single or multi-view.
            file_path = os.path.join(self.shapenet_path, obj, '{}-{}-{}.npy'.format(obj, self.feats_backbone, self.n_views))

            if not os.path.isfile(file_path):
                missing_objs.add(obj)
            else: 
                done_objs.add(obj)

        # This is all the objects. 
        if len(done_objs) >= 7881:
            return

        # Intersect with list of objects actually in dataset. 
        # Note: we can use this same file in both single and multiview, is just a sanity check. 
        snare_objs = get_snare_objs()

        objs = list(snare_objs & missing_objs)
        obj_feats_dict = None

        print('Extracting {} {}-view features for {} objects...'.format(self.feats_backbone, self.n_views, len(objs)))

        # Used to load images quickly. 
        class ObjImgDataset(torch.utils.data.Dataset): 

            def __init__(self, objs, snare_dataset):
                self.objs = objs
                self.snare_dataset = snare_dataset

            def __getitem__(self, idx): 
                item = dict()
                item['images'] = self.snare_dataset.get_imgs(self.objs[idx])
                item['idx'] = idx

                return item

            def __len__(self): 
                return len(self.objs)

        # Now go through objects to extract backbone features. 
        obj_dataset = ObjImgDataset(objs, self)
        bz = 8
        dataloader = torch.utils.data.DataLoader(obj_dataset, batch_size=bz, num_workers=32)

        for b_idx, batch in enumerate(tqdm.tqdm(dataloader)): 
            
            # Get backbone features and prep for input to LegoFormer view embedder (which we'll finetune). 
            imgs = batch['images']

            with torch.no_grad(): 

                if self.n_views > 1:  
                    backbone_feats = model.get_intermediate_feats(imgs) 
                    backbone_feats = backbone_feats.view(imgs.size(0), -1)
                else: 
                    # Pass each image as its own data example. 
                    backbone_feats = model.get_intermediate_feats(imgs.view(imgs.size(0) * 8, 1, 3, 224, 224))
                    backbone_feats = backbone_feats.view(backbone_feats.size(0), -1)
                    backbone_feats = backbone_feats.view(imgs.size(0), 8, backbone_feats.size(-1))

                # TODO need single view case! Can probably just squash over batch dimension and treat each img as its own datapoint. 

            # Now iterate over all objects in batch to store them. 
            for i in range(backbone_feats.size(0)):
                obj_id = obj_dataset.objs[batch['idx'][i]]
                feats = backbone_feats[i].detach().cpu().numpy()
                npy_path = os.path.join(self.shapenet_path, obj_id, '{}-{}-{}.npy'.format(obj_id, self.feats_backbone, self.n_views))

                # Store as npz file. 
                np.save(npy_path, feats) 

    def preprocess_vgg16(self, legoformer_model): 

        # First make set of all objects that don't yet have a feature matrix. 
        missing_objs = set()
        done_objs = set()

        for obj in os.listdir(self.shapenet_path): 
            
            # If skipping legoformer, collect raw vgg16 features altogether without using legoformer. 
            if self.cfg['transformer']['skip_legoformer']: 
                file_path = os.path.join(self.shapenet_path, obj, '{}-rawVGG.npy'.format(obj))

            # Single or multi-view. 
            elif self.n_views == 1: 
                file_path = os.path.join(self.shapenet_path, obj, '{}-single.npy'.format(obj))
            else: 
                file_path = os.path.join(self.shapenet_path, obj, '{}.npy'.format(obj))

            if not os.path.isfile(file_path):
                missing_objs.add(obj)
            else: 
                done_objs.add(obj)

        # This is all the objects. 
        if len(done_objs) >= 7881:
            return

        # Intersect with list of objects actually in dataset. 
        # Note: we can use this same file in both single and multiview, is just a sanity check. 
        snare_objs = get_snare_objs()

        objs = list(snare_objs & missing_objs)
        obj_feats_dict = None

        print('Extracting VGG16 features for {} objects...'.format(len(objs)))

        # Used to load images quickly. 
        class ObjImgDataset(torch.utils.data.Dataset): 

            def __init__(self, objs, snare_dataset):
                self.objs = objs
                self.snare_dataset = snare_dataset

            def __getitem__(self, idx): 
                item = dict()
                item['images'] = self.snare_dataset.get_imgs(self.objs[idx])
                item['idx'] = idx

                return item

            def __len__(self): 
                return len(self.objs)

        # Now go through objects to extract backbone features. 
        obj_dataset = ObjImgDataset(objs, self)
        bz = 8
        dataloader = torch.utils.data.DataLoader(obj_dataset, batch_size=bz, num_workers=32)

        # If skipping LegoFormer, then just use pre-trained VGG16 itself (same as used by LegoFormer though). 
        if self.cfg['transformer']['skip_legoformer']:
            vgg16 = torchvision.models.vgg16(pretrained=True)
            vgg16.classifier = vgg16.classifier[:-1]         
            vgg16.cuda()

        for b_idx, batch in enumerate(tqdm.tqdm(dataloader)): 
            
            # Get backbone features and prep for input to LegoFormer view embedder (which we'll finetune). 
            imgs = batch['images']

            with torch.no_grad(): 

                if self.cfg['transformer']['skip_legoformer']:
                    backbone_feats = imgs.view(imgs.size(0) * 8, 3, 224, 224).cuda()
                    backbone_feats = vgg16(backbone_feats).view(imgs.size(0), 8, -1)

                # Single-view processing. 
                elif self.n_views == 1: 
                    imgs = imgs.view(imgs.size(0) * 8, 1, 3, 224, 224)
                    feats = legoformer_model.legoformer.backbone(imgs)
                    patches = legoformer_model.legoformer.split_features(feats)
                    patches = legoformer_model.legoformer.add_2d_pos_enc(patches)
                    backbone_feats = rearrange(patches, 'b n np d -> b (n np) d')
                    backbone_feats = backbone_feats.reshape(imgs.size(0) // 8, 8, backbone_feats.size(1), backbone_feats.size(-1))
                else:
                    backbone_feats = legoformer_model.legoformer.backbone(imgs)
                    backbone_feats = rearrange(backbone_feats, 'b n c h w -> b n (c h w)')

            # Now iterate over all objects in batch to store them. 
            for i in range(backbone_feats.size(0)):
                obj_id = obj_dataset.objs[batch['idx'][i]]
                feats = backbone_feats[i].detach().cpu().numpy()

                if self.cfg['transformer']['skip_legoformer']:
                    npy_path = os.path.join(self.shapenet_path, obj_id, '{}-rawVGG.npy'.format(obj_id))

                    assert feats.shape == (8, 4096)

                # Single or multiview case.
                elif self.n_views == 1: 
                    npy_path = os.path.join(self.shapenet_path, obj_id, '{}-single.npy'.format(obj_id))
                else: 
                    npy_path = os.path.join(self.shapenet_path, obj_id, '{}.npy'.format(obj_id))

                # Store as npz file. 
                np.save(npy_path, feats) 

    def load_entries(self):
        train_train_files = ["train.json"]
        train_val_files = ["val.json"]
        test_test_files = ["test.json"]

        # modes
        if self.mode == "train":
            self.files = train_train_files
        elif self.mode  == 'valid':
            self.files = train_val_files
        elif self.mode == "test":
            self.files =  test_test_files
        else:
            raise RuntimeError('mode not recognized, should be train, valid or test: ' + str(self.mode))

        # load amt data
        self.data = []
        for file in self.files:
            fname_rel = os.path.join(self.folds, file)
            print(fname_rel)
            with open(fname_rel, 'r') as f:
                self.data = self.data + json.load(f)

        print(f"Loaded Entries. {self.mode}: {len(self.data)} entries")

    def unpack_clip_img_feats(self):
        """
        Load CLIP features to drive in per-object files so that we can use more DataLoader workers. 
        """

        # Load the single feature file itself. 
        with open(self.cfg['data']['clip_img_feats'], 'r') as f:     
            img_feats = json.load(f)

        # Get all unique object IDs and the number of views. 
        object_ids = set()
        view_ids = set()

        for view_key in img_feats.keys(): 
            obj, view = view_key.split('-')

            object_ids.add(obj)
            view_ids.add(int(view))

        # Get ordered list of views for feature ordering. 
        views = list(view_ids)
        views.sort()

        # Store features for all views for each object into a numpy array. 
        for obj_id in tqdm(object_ids): 

            # Will hold CLIP feature. 
            obj_feats = np.zeros((len(views), 512))

            # Iterate over views to get pertinent features for object views. 
            for view in views: 

                # Load feature and store it. 
                obj_view_name = '{}-{}'.format(obj_id, view)
                obj_feats[view] = np.asarray(img_feats[obj_view_name])

            # Write numpy array to disk. 
            feat_path = os.path.join(self.cfg['data']['clip_img_feat_dir'], '{}.npy'.format(obj_id))
            np.save(feat_path, obj_feats)

    def load_extracted_features(self):

        # Determine which features to use. 
        model_type = self.cfg['train']['model']
        self.use_lang_feats = not (model_type == 'transformer' or model_type == 'pixelnerf')
        self.use_img_feats = True# TODO = self.feats_backbone == "clip" or self.feats_backbone == 'multimodal' 
        
        # Load pre-trained CLIP language features if not using transformer model.  
        if self.use_lang_feats: 
            lang_feats_path = self.cfg['data']['clip_lang_feats']
            with open(lang_feats_path, 'r') as f:
                self.lang_feats = json.load(f)

        # Make sure img features have been unpacked to disk if using them.   
        if self.use_img_feats: 

            # Unpack and store if needed. 
            self.clip_img_feat_dir = self.cfg['data']['clip_img_feat_dir']

            if not os.path.isdir(self.clip_img_feat_dir):
                os.mkdir(self.clip_img_feat_dir)
                
                self.unpack_clip_img_feats()

    def __len__(self):
        # Accomodate for larger dataset if different combos are possible. 
        if self.n_views != 8 and self.mode != 'train': 
            return len(self.data) * 10
        else: 
            return len(self.data)

    def get_img_feats(self, key):

        # Load image features for object. 
        feat_path = os.path.join(self.clip_img_feat_dir, '{}.npy'.format(key))
        feats = np.load(feat_path)

        # Return all the desired views. 
        return feats[np.arange(self.total_views)]

    def get_obj_feats(self, obj): 
        file_path = os.path.join(self.shapenet_path, obj, '{}-{}-{}.npy'.format(obj, self.feats_backbone, self.n_views))

        return np.load(file_path)

    def get_imgs(self, key): 
        
        # Object images path. 
        img_dir = os.path.join(self.img_path, key)

        # Iterate over images and load them. 
        imgs = []
        img_idxs = np.arange(self.total_views)[6:]# TODO we hardcode the standard 8-views for now. 

        for idx in img_idxs: 

            # Standard and custom renders have different paths. 
            if self.cfg['data']['custom_renders']:
                
                # Name based on degree of rotation. 
                degree = "{:03d}".format((idx - 6) * 45)
                img_path = os.path.join(img_dir, '{}.png'.format(degree))

                # Load image and set white background to fit PixelNeRF distribution. 
                img = Image.open(img_path).convert("RGBA")
                background = Image.new('RGBA', img.size, (255,255,255))
                alpha_composite = Image.alpha_composite(background, img)
                alpha_composite_3 = alpha_composite.convert('RGB')
                img = np.asarray(alpha_composite_3).astype(np.float32) / 255.0
            else: 
                img_path =  os.path.join(img_dir, '{}-{}.png'.format(key, idx))            
                img = ShapeNetDataset.read_img(img_path)
                img = cv2.resize(img, (64, 64))

            imgs.append(img)

        imgs = np.asarray(imgs)

        # Add transformations from LegoFormer. THIS STEP IS CRUCIAL.
        if self.transforms: 
            imgs = self.transforms(imgs)
        else: 
            # Change ordering of dimensions and get rid of alpha channel. 
            imgs = np.transpose(imgs, (0, 3, 1, 2))[:,:3,:,:]

        return imgs

    def get_vgg16_feats(self, key): 

        # Object images path. 
        feat_dir = os.path.join(self.shapenet_path, key)

        if self.cfg['transformer']['skip_legoformer']:
            feat_path = os.path.join(feat_dir, '{}-rawVGG.npy'.format(key))

        # Single-view features. 
        elif self.n_views == 1: 
            feat_path = os.path.join(feat_dir, '{}-single.npy'.format(key))
        
        # Multi-view features. 
        else: 
            feat_path = os.path.join(feat_dir, '{}.npy'.format(key))
        
        return np.load(feat_path)

    def __getitem__(self, idx):

        if self.cfg['train']['tiny_dataset']:
            idx = idx % self.cfg['train']['batch_size']
        
        # Add more examples for stability if different view combinations are possible. 
        if self.n_views != 8 and self.mode != 'train': 
            idx = idx // 10

        # Will return features in dictionary form. 
        feats = dict()
        entry = self.data[idx]

        # get keys
        entry_idx = entry['ans'] if 'ans' in entry else -1 # test set does not contain answers
        if len(entry['objects']) == 2:
            key1, key2 = entry['objects']

        # fix missing key in pair by sampling alternate different object from data. 
        else:
            key1 = entry['objects'][entry_idx]
 
            while True:

                alt_entry = self.data[np.random.choice(len(self.data))]
                key2 = np.random.choice(alt_entry['objects'])

                if key2 != key1:
                    break
        
        # annotation
        annotation = entry['annotation']
        feats['annotation'] = annotation

        # test set does not have labels for visual and non-visual categories
        feats['is_visual'] = entry['visual'] if 'ans' in entry else -1

        # Select view indexes randomly # TODO need to select them consistently for evaluation.
        view_idxs1 = np.random.choice(8, self.n_views, replace=False)
        view_idxs2 = np.random.choice(8, self.n_views, replace=False)

        ## Img feats
        # For CLIP filter to use only desired amount of views (in this case 8). 
        if self.use_img_feats: 
            start_idx = 6 # discard first 6 views that are top and bottom viewpoints
            img1_n_feats = torch.from_numpy(self.get_img_feats(key1))[start_idx:]
            img2_n_feats = torch.from_numpy(self.get_img_feats(key2))[start_idx:]
      
            # Pick out sampled views. 
            img1_n_feats = img1_n_feats[view_idxs1]
            img2_n_feats = img2_n_feats[view_idxs2]

            feats['img_feats'] = (img1_n_feats, img2_n_feats)

        # Object reconstruction model features, except for legoformer. 
        if self.feats_backbone == 'pix2vox' or self.feats_backbone == '3d-r2n2':
            obj1_n_feats = torch.from_numpy(self.get_obj_feats(key1))
            obj2_n_feats = torch.from_numpy(self.get_obj_feats(key2))
           
            if self.n_views == 1: 
                obj1_n_feats = obj1_n_feats[view_idxs1]
                obj2_n_feats = obj2_n_feats[view_idxs2]

            feats['obj_feats'] = (obj1_n_feats, obj2_n_feats)

        # Tokenize annotation if using a transformer.
        use_lang_toks = self.cfg['train']['model'] == 'transformer' or \
                self.cfg['train']['model'] == 'pixelnerf'

        if use_lang_toks:
            feats['lang_tokens'] = clip.tokenize(feats['annotation'])
        else: 
            feats['lang_feats'] = torch.from_numpy(np.array(self.lang_feats[annotation]))

        # label
        feats['ans'] = entry_idx
    
        # Keys
        feats['keys'] = (key1, key2)

        # Return VGG16 features if needed.
        if self.feats_backbone == 'legoformer': 
            vgg16_feats1 = self.get_vgg16_feats(key1)
            vgg16_feats2 = self.get_vgg16_feats(key2)

            # Filter out views. 
            vgg16_feats1 = vgg16_feats1[view_idxs1]
            vgg16_feats2 = vgg16_feats2[view_idxs2]

            feats['vgg16_feats'] = (vgg16_feats1, vgg16_feats2)

        # Load ground truth voxel maps if needed. 
        if self.cfg['data']['voxel_reconstruction']:
            volume1_path = os.path.join(self.cfg['data']['shapenet_voxel_dir'], '{}-32.npy'.format(key1))
            volume2_path = os.path.join(self.cfg['data']['shapenet_voxel_dir'], '{}-32.npy'.format(key2))

            # Skip voxel maps we don't have good data for. 
            if os.path.isfile(volume1_path):
                volume1 = np.load(volume1_path)
                volume1_mask = 1.0
            else: 
                volume1 = np.zeros((32, 32, 32), dtype=np.float32)
                volume1_mask = 0.0

            if os.path.isfile(volume2_path):
                volume2 = np.load(volume2_path)
                volume2_mask = 1.0
            else: 
                volume2 = np.zeros((32, 32, 32), dtype=np.float32)
                volume2_mask = 0.0

            feats['voxel_maps'] = (volume1, volume2)
            feats['voxel_masks'] = (volume1_mask, volume2_mask)

        # Get pixelnerf features if using them. 
        if self.cfg['train']['model'] == 'pixelnerf':
            
            # Get features for object. 
            obj1_feats = np.load(os.path.join(self.pixelnerf_feat_dir, '{}.npy'.format(key1)))
            obj2_feats = np.load(os.path.join(self.pixelnerf_feat_dir, '{}.npy'.format(key2)))

            # Get only the views requested. 
            obj1_feats = obj1_feats[view_idxs1]
            obj2_feats = obj2_feats[view_idxs2]

            # Get camera parameters for views. 
            obj1_cam = self.cam_poses[view_idxs1]
            obj2_cam = self.cam_poses[view_idxs2]

            feats['obj_feats'] = (obj1_feats, obj2_feats)
            feats['obj_cams'] = (obj1_cam, obj2_cam)

        return feats
