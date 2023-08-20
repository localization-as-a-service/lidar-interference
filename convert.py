import os
import tqdm
import cv2
import numpy as np

def make_video(sequence_dir, output_path):
    sequence_ts = os.listdir(sequence_dir)
    # print(sequence_ts)
    sequence_ts = [int(sequence_ts[i].split(".")[0].split("-")[1]) for i in range(len(sequence_ts)) if sequence_ts[i].endswith(".depth.png")]
    sequence_ts = sorted(sequence_ts)
    images = []
 
    
    for t in sequence_ts:
        img = cv2.imread(os.path.join(sequence_dir, f"frame-{t}.depth.png"), cv2.IMREAD_ANYDEPTH)
        img = img / (9 * 4000)
        img = img * 255
        img = np.where(img > 255, 255, img)
        img = np.array(img, dtype=np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        images.append(img)
 
    video = cv2.VideoWriter(f"{output_path}", cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 480))
    for image in tqdm.tqdm(images):
        video.write(image)

data_dir = "data/raw_data/exp_1"

for trial in os.listdir(data_dir):
    for subject in os.listdir(os.path.join(data_dir, trial, "secondary")):
        for sequence in os.listdir(os.path.join(data_dir, trial, "secondary", subject)):
            sequence_dir = os.path.join(data_dir, trial, "secondary", subject, sequence, "frames")
            output_path = os.path.join("data/videos/exp_1", f"{trial}_{subject}_{sequence}.avi")
            
            if os.path.exists(output_path):
                continue
            
            print(sequence_dir)
            make_video(sequence_dir, output_path)

# for trial in os.listdir(data_dir):
#     if not trial.startswith("trial_"):
#         continue
#     for device in os.listdir(os.path.join(data_dir, trial, "global")):
#         if not device.startswith("device"):
#             continue
        
#         sequence_dir = os.path.join(data_dir, trial, "global", device)
#         output_path = os.path.join("output", f"{trial}_{device}.avi")
        
#         if os.path.exists(output_path):
#             continue
        
#         print(sequence_dir)
        
#         make_video(sequence_dir, output_path)
        
        