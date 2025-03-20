import torch
from torch.utils.data import Dataset
from .build import DATASETS
import numpy as np
import pandas as pd
import laspy
import os
from pathlib import Path
from tqdm import tqdm

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

        self.processed_dir = os.path.join(self.root, f'processed_{self.npoints}pts')
        self.save_path = os.path.join(
            self.processed_dir, 
            f'forest_species_{self.subset}_{self.npoints}pts.npz'
        )

        self.file_paths, self.labels = self._collect_files(base_folder)
        self._process_or_load_data()

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

    def _collect_files(self, base_folder):
        file_paths = []
        labels = []
        
        for _, row in self.metadata.iterrows():
            meta_filename = row['filename'].lstrip('/')
            meta_filename = meta_filename.replace('.las', '.laz')
            filename = os.path.basename(meta_filename)
            laz_path = os.path.join(self.root, base_folder, filename)
            
            if os.path.exists(laz_path):
                file_paths.append(laz_path)
                labels.append(self.species_to_idx[row['species']])
            else:
                las_path = laz_path.replace('.laz', '.las')
                if os.path.exists(las_path):
                    file_paths.append(las_path)
                    labels.append(self.species_to_idx[row['species']])
        
        if not file_paths:
            existing_files = os.listdir(os.path.join(self.root, base_folder))
            raise RuntimeError(f"No valid files found in {os.path.join(self.root, base_folder)}")
        
        return file_paths, labels

    def _process_or_load_data(self):
        if os.path.exists(self.save_path):
            print(f'Loading preprocessed data from {self.save_path}')
            loaded = np.load(self.save_path)
            self.processed_points = loaded['points']
            self.processed_labels = loaded['labels']
        else:
            print(f'Starting preprocessing of point cloud data...')
            os.makedirs(self.processed_dir, exist_ok=True)
            
            num_files = len(self.file_paths)
            self.processed_points = np.zeros((num_files, self.npoints, 3), dtype=np.float32)
            self.processed_labels = np.array(self.labels, dtype=np.int64)
            
            with tqdm(total=num_files, desc='Processing point clouds', 
                    unit='files', ncols=80) as pbar:
                for idx in range(num_files):
                    points = self.load_laz_file(self.file_paths[idx])
                    self.processed_points[idx] = points
                    
                    current_file = os.path.basename(self.file_paths[idx])
                    pbar.set_postfix({'file': current_file}, refresh=True)
                    pbar.update(1)
            
            print(f'\nSaving processed data to {self.save_path}')
            np.savez(
                self.save_path,
                points=self.processed_points,
                labels=self.processed_labels
            )
            print('Preprocessing complete!')

    def __getitem__(self, idx):
        points = torch.from_numpy(self.processed_points[idx]).float()
        label = self.processed_labels[idx]
        return points, label

    def __len__(self):
        return len(self.processed_points)
    
    @property
    def class_names(self):
        return [self.idx_to_species[i] for i in range(self.num_classes)]