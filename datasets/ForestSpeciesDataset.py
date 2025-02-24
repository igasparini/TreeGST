import torch
from torch.utils.data import Dataset
from .build import DATASETS
import numpy as np
import pandas as pd
import laspy
import os
from pathlib import Path

def pc_normalize(pc):
    """Normalize point cloud to unit sphere centered at origin"""
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

@DATASETS.register_module()
class ForestSpecies(Dataset):
    def __init__(self, config):
        self.root = config.DATA_PATH
        self.subset = config.subset
        self.npoints = config.N_POINTS
        metadata_path = os.path.join(self.root, 'tree_metadata_dev.csv')
        self.metadata = pd.read_csv(metadata_path)
        
        # species to index mapping
        unique_species = sorted(self.metadata['species'].unique())
        self.species_to_idx = {species: idx for idx, species in enumerate(unique_species)}
        self.idx_to_species = {idx: species for species, idx in self.species_to_idx.items()}
        self.num_classes = len(self.species_to_idx)
        
        if self.subset == 'train':
            base_folder = os.path.join('dev', 'train')
        else:
            base_folder = 'test'

        self.file_paths = []
        self.labels = []

        for _, row in self.metadata.iterrows():
            meta_filename = row['filename'].lstrip('/')
            meta_filename = meta_filename.replace('.las', '.laz')
            filename = os.path.basename(meta_filename)
            laz_path = os.path.join(self.root, base_folder, filename)
            
            if os.path.exists(laz_path):
                self.file_paths.append(laz_path)
                self.labels.append(self.species_to_idx[row['species']])
            else:
                print(f"Warning: File not found - {laz_path}")
                las_path = laz_path.replace('.laz', '.las')
                if os.path.exists(las_path):
                    self.file_paths.append(las_path)
                    self.labels.append(self.species_to_idx[row['species']])
                    print(f"Found .las file instead: {las_path}")
        
        if len(self.file_paths) == 0:
            existing_files = os.listdir(os.path.join(self.root, base_folder))
            raise RuntimeError(
                f"No valid files found in {os.path.join(self.root, base_folder)}\n"
                f"Existing files: {existing_files[:5]}\n"
            )
        
        print(f'Successfully loaded {len(self.file_paths)} samples from {base_folder}')

    def load_laz_file(self, file_path):
        try:
            las = laspy.read(file_path, laz_backend=laspy.LazBackend.Lazrs)
            points = np.vstack((las.x, las.y, las.z)).transpose()
            points = pc_normalize(points)
            
            if len(points) < self.npoints:
                # repeat points if less than N_POINTS
                repeat_factor = int(np.ceil(self.npoints / len(points)))
                points = np.tile(points, (repeat_factor, 1))
            
            points = farthest_point_sample(points, self.npoints)
            return points.astype(np.float32)
            
        except Exception as e:
            raise RuntimeError(f"Error loading file {file_path}: {str(e)}")

    def get_species_name(self, idx):
        return self.idx_to_species[idx]

    def __getitem__(self, idx):          
        points = self.load_laz_file(self.file_paths[idx])
        label = self.labels[idx]
        points = torch.from_numpy(points).float()
        return points, label

    def __len__(self):
        return len(self.file_paths)

    @property
    def class_names(self):
        return [self.idx_to_species[i] for i in range(self.num_classes)]