import os
import open3d
import numpy as np
import tqdm

import utils.pointcloud as pointcloud
import utils.fread as fread
import utils.functions as functions

from utils.depth_camera import DepthCamera


def convert_to_point_clouds(dataset_dir, subject_id=1, device_id=3, aligned=True):
    """
    Go through the dataset structure and convert all the depth images to point clouds

    Args:
        dataset_dir (str): the directory contains the raw captures
    """
    
    out_dir = dataset_dir.replace("raw_data", "point_clouds")

    if not os.path.exists(out_dir): os.makedirs(out_dir)

    # Secondary camera
    device = DepthCamera("device-3", os.path.join(dataset_dir, f"../metadata/device-{device_id}-aligned.json"))

    # Iterate through the secondary directory
    subject = f"subject-{subject_id}"
    for seq_id in os.listdir(os.path.join(dataset_dir, "secondary", subject)):
        sequence_dir = os.path.join(dataset_dir, "secondary", subject, seq_id, "frames")
        seq_out_dir = os.path.join(out_dir, subject, seq_id)

        if not os.path.exists(seq_out_dir): os.makedirs(seq_out_dir)

        sequence_ts = fread.get_timstamps_from_images(sequence_dir, ".depth.png")
        # sequence_ts = fread.sample_timestamps(sequence_ts, 20)

        for t in tqdm.tqdm(sequence_ts):
            if os.path.exists(os.path.join(seq_out_dir, f"{t}.secondary.pcd")):
                continue
            
            secondary_pcd = device.depth_to_point_cloud(os.path.join(sequence_dir, f"frame-{t}.depth.png"))
            secondary_pcd = open3d.geometry.voxel_down_sample(secondary_pcd, voxel_size=0.03)
            
            open3d.io.write_point_cloud(os.path.join(seq_out_dir, f"{t}.secondary.pcd"), secondary_pcd)


if __name__ == "__main__":
    convert_to_point_clouds("data/raw_data/exp_1/trial_1", subject_id=2, device_id=3, aligned=False)
