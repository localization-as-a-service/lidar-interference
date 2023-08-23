import os
import glob
import open3d
import numpy as np
import tqdm

from utils.depth_camera import DepthCamera


def convert_to_point_clouds(dataset_dir, device_id=0):
    out_dir = dataset_dir.replace("raw_data", "point_clouds")
    
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    
    device = DepthCamera("device", os.path.join(dataset_dir, f"metadata/device-{device_id}-aligned.json"))
    
    depth_img_files = glob.glob(os.path.join(dataset_dir, "*.depth.png"))
    
    for depth_img_file in tqdm.tqdm(depth_img_files):
        pcd = device.depth_to_point_cloud(depth_img_file)
        file_name = os.path.basename(depth_img_file).replace("depth.png", "pcd")
        
        open3d.io.write_point_cloud(os.path.join(out_dir, file_name), pcd)
        
if __name__ == "__main__":
    convert_to_point_clouds("data/raw_data/intel_l515/exp_3", 4)
    
    