# Voxelizer modified from ARM for DDP training
# Source: https://github.com/stepjam/ARM
# License: https://github.com/stepjam/ARM/LICENSE

from functools import reduce
from operator import mul
import pdb

import torch
from torch import nn
from torch.utils.data import DataLoader
import open3d as o3d
import numpy as np
import os
from tqdm import tqdm

MIN_DENOMINATOR = 1e-12
INCLUDE_PER_VOXEL_COORD = False


class VoxelGrid(nn.Module):

    def __init__(self,
                 coord_bounds,
                 voxel_size: int,
                 device,
                 batch_size,
                 feature_size,  # e.g. rgb or image features
                 max_num_coords: int,):
        super(VoxelGrid, self).__init__()
        self._device = device
        self._voxel_size = voxel_size
        self._voxel_shape = [voxel_size] * 3
        self._voxel_d = float(self._voxel_shape[-1])
        self._voxel_feature_size = 4 + feature_size
        self._voxel_shape_spec = torch.tensor(self._voxel_shape,
                                              ).unsqueeze(
            0) + 2  # +2 because we crop the edges.
        self._coord_bounds = torch.tensor(coord_bounds, dtype=torch.float,
                                          ).unsqueeze(0)
        max_dims = self._voxel_shape_spec[0]
        self._total_dims_list = torch.cat(
            [torch.tensor([batch_size], ), max_dims,
             torch.tensor([4 + feature_size], )], -1).tolist()

        self.register_buffer('_ones_max_coords', torch.ones((batch_size, max_num_coords, 1)))
        self._num_coords = max_num_coords

        shape = self._total_dims_list
        result_dim_sizes = torch.tensor(
            [reduce(mul, shape[i + 1:], 1) for i in range(len(shape) - 1)] + [1], )
        self.register_buffer('_result_dim_sizes', result_dim_sizes)
        flat_result_size = reduce(mul, shape, 1)

        self._initial_val = torch.tensor(0, dtype=torch.float)
        flat_output = torch.ones(flat_result_size, dtype=torch.float) * self._initial_val
        self.register_buffer('_flat_output', flat_output)

        self.register_buffer('_arange_to_max_coords', torch.arange(4 + feature_size))
        self._flat_zeros = torch.zeros(flat_result_size, dtype=torch.float)

        self._const_1 = torch.tensor(1.0, )
        self._batch_size = batch_size

        # Coordinate Bounds:
        bb_mins = self._coord_bounds[..., 0:3]
        self.register_buffer('_bb_mins', bb_mins)
        bb_maxs = self._coord_bounds[..., 3:6]
        bb_ranges = (bb_maxs - bb_mins).cuda()
        # get voxel dimensions. 'DIMS' mode
        self._dims = dims = (self._voxel_shape_spec.int()).cuda()
        dims_orig = (self._voxel_shape_spec.int() - 2).cuda()
        self.register_buffer('_dims_orig', dims_orig)

        # self._dims_m_one = (dims - 1).int()
        dims_m_one = (dims - 1).int()
        self.register_buffer('_dims_m_one', dims_m_one)

        # BS x 1 x 3
        res = bb_ranges / (dims_orig.float() + MIN_DENOMINATOR)
        self._res_minis_2 = bb_ranges / (dims.float() - 2 + MIN_DENOMINATOR)
        self.register_buffer('_res', res)

        voxel_indicy_denmominator = res + MIN_DENOMINATOR
        self.register_buffer('_voxel_indicy_denmominator', voxel_indicy_denmominator)

        self.register_buffer('_dims_m_one_zeros', torch.zeros_like(dims_m_one))

        batch_indices = torch.arange(self._batch_size, dtype=torch.int).view(self._batch_size, 1, 1)
        self.register_buffer('_tiled_batch_indices', batch_indices.repeat([1, self._num_coords, 1]))

        w = self._voxel_shape[0] + 2
        arange = torch.arange(0, w, dtype=torch.float, )
        index_grid = torch.cat([
            arange.view(w, 1, 1, 1).repeat([1, w, w, 1]),
            arange.view(1, w, 1, 1).repeat([w, 1, w, 1]),
            arange.view(1, 1, w, 1).repeat([w, w, 1, 1])], dim=-1).unsqueeze(
            0).repeat([self._batch_size, 1, 1, 1, 1])
        self.register_buffer('_index_grid', index_grid)

    def _broadcast(self, src: torch.Tensor, other: torch.Tensor, dim: int):
        if dim < 0:
            dim = other.dim() + dim
        if src.dim() == 1:
            for _ in range(0, dim):
                src = src.unsqueeze(0)
        for _ in range(src.dim(), other.dim()):
            src = src.unsqueeze(-1)
        src = src.expand_as(other)
        return src

    def _scatter_mean(self, src: torch.Tensor, index: torch.Tensor, out: torch.Tensor,
                      dim: int = -1):
        out = out.scatter_add_(dim, index, src)

        index_dim = dim
        if index_dim < 0:
            index_dim = index_dim + src.dim()
        if index.dim() <= index_dim:
            index_dim = index.dim() - 1

        ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
        out_count = torch.zeros(out.size(), dtype=out.dtype, device=out.device)
        out_count = out_count.scatter_add_(index_dim, index, ones)
        out_count.clamp_(1)
        count = self._broadcast(out_count, out, dim)
        if torch.is_floating_point(out):
            out.true_divide_(count)
        else:
            out.floor_divide_(count)
        return out

    def _scatter_nd(self, indices, updates):
        indices_shape = indices.shape
        num_index_dims = indices_shape[-1]
        flat_updates = updates.view((-1,))
        indices_scales = self._result_dim_sizes[0:num_index_dims].view(
            [1] * (len(indices_shape) - 1) + [num_index_dims])
        indices_for_flat_tiled = ((indices * indices_scales).sum(
            dim=-1, keepdims=True)).view(-1, 1).repeat(
            *[1, self._voxel_feature_size])

        implicit_indices = self._arange_to_max_coords[
                           :self._voxel_feature_size].unsqueeze(0).repeat(
            *[indices_for_flat_tiled.shape[0], 1])
        indices_for_flat = indices_for_flat_tiled + implicit_indices
        flat_indices_for_flat = indices_for_flat.view((-1,)).long()

        flat_scatter = self._scatter_mean(
            flat_updates, flat_indices_for_flat,
            out=torch.zeros_like(self._flat_output))
        return flat_scatter.view(self._total_dims_list)

    def coords_to_bounding_voxel_grid(self, coords, coord_features=None,
                                      coord_bounds=None):
        
        voxel_indicy_denmominator = self._voxel_indicy_denmominator
        res, bb_mins = self._res, self._bb_mins
        if coord_bounds is not None:
            bb_mins = coord_bounds[..., 0:3]
            bb_maxs = coord_bounds[..., 3:6]
            bb_ranges = bb_maxs - bb_mins
            res = bb_ranges / (self._dims_orig.float() + MIN_DENOMINATOR)
            voxel_indicy_denmominator = res + MIN_DENOMINATOR

        bb_mins_shifted = bb_mins - res  # shift back by one
        floor = torch.floor(
            (coords - bb_mins_shifted.unsqueeze(1)) / voxel_indicy_denmominator.unsqueeze(1)).int()
        voxel_indices = torch.min(floor, self._dims_m_one)
        voxel_indices = torch.max(voxel_indices, self._dims_m_one_zeros)

        # BS x NC x 3
        voxel_values = coords
        if coord_features is not None:
            voxel_values = torch.cat([voxel_values, coord_features], -1)

        _, num_coords, _ = voxel_indices.shape
        # BS x N x (num_batch_dims + 2)
        all_indices = torch.cat([
            self._tiled_batch_indices[:, :num_coords], voxel_indices], -1)

        # BS x N x 4
        voxel_values_pruned_flat = torch.cat(
            [voxel_values, self._ones_max_coords[:, :num_coords]], -1)

        # BS x x_max x y_max x z_max x 4
        scattered = self._scatter_nd(
            all_indices.view([-1, 1 + 3]),
            voxel_values_pruned_flat.view(-1, self._voxel_feature_size))

        vox = scattered[:, 1:-1, 1:-1, 1:-1]
        if INCLUDE_PER_VOXEL_COORD:
            res_expanded = res.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            res_centre = (res_expanded * self._index_grid) + res_expanded / 2.0
            coord_positions = (res_centre + bb_mins_shifted.unsqueeze(
                1).unsqueeze(1).unsqueeze(1))[:, 1:-1, 1:-1, 1:-1]
            vox = torch.cat([vox[..., :-1], coord_positions, vox[..., -1:]], -1)

        occupied = (vox[..., -1:] > 0).float()
        vox = torch.cat([
            vox[..., :-1], occupied], -1)

        return torch.cat(
           [vox[..., :-1], self._index_grid[:, :-2, :-2, :-2] / self._voxel_d,
            vox[..., -1:]], -1)
        
class RGBPCDataset(torch.utils.data.Dataset):
            
    def __init__(self, pcd_folder):
        self.pcd_folder = pcd_folder
        
        # Get all file keys. 
        files = os.listdir(self.pcd_folder)
        self.keys = [f.split('.')[0] for f in files if f.endswith('.pcd')]
        
        # Folder to put all processed voxelmaps into. 
        parent_dir = os.path.split(self.pcd_folder)[0]
        self.vm_folder = os.path.join(parent_dir, 'shapenetsem_rgb_voxelmaps')
            
        # Create folder if it doesn't exist.
        if not os.path.isdir(self.vm_folder):
            os.mkdir(self.vm_folder)
            
    def __len__(self):
        return len(self.keys)
            
    def load_pointcloud(self, key):
        
        def pc_norm(pc):
            """ pc: NxC, return NxC """

            # first normalize it between 0 and 1!
            pc = pc

            centroid = np.mean(pc, axis=0)
            pc = pc - centroid
            m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
            pc = pc / m
            return pc
        
        # Load point cloud. 
        pc = o3d.io.read_point_cloud(os.path.join(self.pcd_folder, '{}.pcd'.format(key)))
        data = np.array(pc.points).astype(np.float32)
        colors = np.array(pc.colors).astype(np.float32)

        # Normalize colors.
        if (np.amax(colors) - np.amin(colors)) != 0:
            normalized_colors = (colors - np.amin(colors)) / (np.amax(colors) - np.amin(colors))
        else: 
            normalized_colors = (colors - np.amin(colors))

        colors = 2*normalized_colors-1

        # Normalize point cloud.
        data = pc_norm(data)

        # Concatenate colors to point cloud.
        data = np.concatenate([data, colors], -1)
        data = torch.from_numpy(data).float()

        return data
    
    def __getitem__(self, idx):
        
        # Load rgb pcd. 
        pcd = self.load_pointcloud(self.keys[idx])

        return pcd, idx
    
if __name__ == '__main__':
    
    # Dataset
    dataset = RGBPCDataset(pcd_folder='/home/rcorona/2022/lang_nerf/vlg/snare-master/data/shapenetsem_color_pc')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)
    
    # Voxelizer
    bounds = torch.Tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]).cuda()
    
    voxelizer = VoxelGrid(coord_bounds=bounds,
        voxel_size=32,
        device='cuda:0',
        batch_size=1,
        feature_size=3,
        max_num_coords=8912,
    ).cuda()

    # Loop for voxelizing RGB PCs
    for rgbpcd, idx in tqdm(dataloader):
        
        # Voxelize batch. 
        bz = rgbpcd.size(0)
        rgbpcd = rgbpcd.cuda()
        
        pcd = rgbpcd[:,:,:3]
        rgb = rgbpcd[:,:,3:]
        
        voxel_grid = voxelizer.coords_to_bounding_voxel_grid(
            pcd, coord_features=rgb, coord_bounds=bounds)
        voxel_grid = voxel_grid.permute(0, 4, 1, 2, 3).detach().squeeze().cpu().numpy()
    
        # Save voxel grid. 
        idx = idx.item()
        key = dataset.keys[idx]
        save_path = os.path.join(dataset.vm_folder, '{}.npy'.format(key))

        np.save(save_path, voxel_grid)