import open3d
import numpy as np
import pandas as pd
import os
import glob
import tqdm
import copy
import time

import utils.registration as registration
import utils.functions as functions
import utils.transform as transform
import utils.pointcloud as pointcloud
import utils.fread as fread
import utils.FCGF as FCGF

from utils.config import Config
from scipy.ndimage import gaussian_filter1d
from PIL import Image


def fpfh_local_registration(config: Config):
    output_file = config.get_output_file(config.get_file_name() + ".npz")
    
    if os.path.exists(output_file):
        print(f"-- File {output_file} already exists. Skipping.")
        return
    
    pcd_dir = f"data/point_clouds/{config.experiment}/{config.trial}/{config.subject}/{config.sequence}"

    sequence_ts = fread.get_timstamps(pcd_dir, ext=".secondary.pcd")
    sequence_ts = fread.sample_timestamps(sequence_ts, config.target_fps)
    
    num_frames = len(sequence_ts)
    print(f"-- Number of frames: {num_frames}")
    
    print("-- Caching local PCDs and features.")
    
    local_pcds = []
    fpfh_feats = []
    
    start_time = time.time()

    for t in tqdm.trange(num_frames):
        pcd_file = os.path.join(pcd_dir, f"{sequence_ts[t]}.secondary.pcd")
        pcd = open3d.io.read_point_cloud(pcd_file)
        pcd, fpfh = registration.compute_fpfh(pcd, config.voxel_size, down_sample=False, compute_normals=True)
        local_pcds.append(pcd)
        fpfh_feats.append(fpfh) 
        
    end_time = time.time()
    print(f"-- Caching took {end_time - start_time} seconds.")
    print(f"-- Average time per frame: {(end_time - start_time) / num_frames} seconds.")
    print(f"-- Itr/second: {num_frames / (end_time - start_time)}")
        
    print("-- Registering local PCDs.")
    
    local_t = [np.identity(4)]
    start_time = time.time()
    
    for t in tqdm.trange(num_frames - 1):
        source, source_fpfh = copy.deepcopy(local_pcds[t + 1]), fpfh_feats[t + 1]
        target, target_fpfh = copy.deepcopy(local_pcds[t]), fpfh_feats[t]

        reg_result = registration.exec_ransac(source, target, source_fpfh, target_fpfh, n_ransac=4, threshold=0.05)
        reg_result = registration.exec_icp(source, target, threshold=0.05, trans_init=reg_result.transformation, max_iteration=200, p2p=False)

        local_t.append(reg_result.transformation)
    
    end_time = time.time()
    print(f"-- Caching took {end_time - start_time} seconds.")
    print(f"-- Average time per frame: {(end_time - start_time) / num_frames} seconds.")
    print(f"-- Itr/second: {num_frames / (end_time - start_time)}")
    
    print("-- Refining Trajectory.")
    
    trajectory_t = [np.identity(4)]

    for t in tqdm.trange(1, num_frames):
        trajectory_t.append(np.dot(trajectory_t[t - 1], local_t[t]))
        
    print("-- Saving Trajectory.")
    
    np.savez_compressed(output_file, sequence_ts=sequence_ts, local_t=local_t, trajectory_t=trajectory_t)
    
if __name__ == "__main__":
    config = Config(
        sequence_dir="data/raw_data",
        feature_dir="data/features",
        output_dir="data/trajectories/local/FPFH_0.03",
        experiment="exp_1",
        trial="trial_1",
        subject="subject-1",
        sequence="01",
        groundtruth_dir="data/trajectories/groundtruth",
    )
    
    config.voxel_size=0.03
    config.target_fps=20
    config.min_std=0.5
    
    for trial in os.listdir(os.path.join(config.sequence_dir, config.experiment)):
        if not trial.startswith("trial"):
            continue
        config.trial = trial
        for subject in os.listdir(os.path.join(config.sequence_dir, config.experiment, config.trial, "secondary")):
            config.subject = subject    
            for sequence in os.listdir(os.path.join(config.sequence_dir, config.experiment, config.trial, "secondary", config.subject)):
                config.sequence = sequence
                print(f"Processing: {config.experiment} >> {config.trial} >> {config.subject} >> {config.sequence}")
                fpfh_local_registration(config)