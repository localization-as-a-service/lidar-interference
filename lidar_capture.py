import argparse
import os
import time
import cv2
import numpy as np
import pyrealsense2 as rs


def camera_stream(out_dir):
    camera_pipe = rs.pipeline()
    camera_config = rs.config()
    
    camera_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    camera_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    align = rs.align(rs.stream.color)
    
    camera_pipe.start(camera_config)
    
    previous_t = time.time() * 1000
    start_t = time.time()
    elapsed_t = 0
    
    while True:
        try:
            elapsed_t = time.time() - start_t
            
            if elapsed_t > 20:
                break
            
            camera_frames: rs.composite_frame = camera_pipe.poll_for_frames()
            if camera_frames:
                current_t = camera_frames.get_frame_metadata(rs.frame_metadata_value.time_of_arrival)
                aligned_frames = align.process(camera_frames)

                aligned_depth_frame = aligned_frames.get_depth_frame()
                aligned_color_frame = aligned_frames.get_color_frame()

                if not aligned_depth_frame or not aligned_color_frame:
                    continue

                depth_image = np.asanyarray(aligned_depth_frame.get_data())
                color_image = np.asanyarray(aligned_color_frame.get_data())
                
                fps = int(1 / (current_t - previous_t + 1) * 1e3)
                previous_t = current_t

                cv2.imwrite(os.path.join(out_dir, "frames", f"frame-{current_t}.color.png"), color_image)
                cv2.imwrite(os.path.join(out_dir, "frames", f"frame-{current_t}.depth.png"), depth_image)
                
                message = f"FPS: {fps:2d} | Elapsed Time: {elapsed_t:03.2f}s"

                cv2.putText(color_image, message, (7, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 76, 231), 2, cv2.LINE_AA)
                cv2.namedWindow("Secondary View", cv2.WINDOW_AUTOSIZE)
                cv2.imshow("Secondary View", color_image)

                # To get key presses
                key = cv2.waitKey(1)

                if key in (27, ord("q")) or cv2.getWindowProperty("Secondary View", cv2.WND_PROP_AUTOSIZE) < 0:
                    break
                
        except KeyboardInterrupt:
            break

    camera_pipe.stop()
    cv2.destroyAllWindows()
                    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Intel Lidar L515")
    parser.add_argument("--experiment", default=0, type=int)
    parser.add_argument("--trial", default=0, type=int)
    parser.add_argument("--subject", default=0, type=int)
    parser.add_argument("--sequence", default=0, type=int)
    args = parser.parse_args()

    out_dir = f"data/raw_data/exp_{args.experiment}/trial_{args.trial}/secondary/subject-{args.subject}/{args.sequence:02d}"
    
    if not os.path.exists(os.path.join(out_dir, "frames")):
        os.makedirs(os.path.join(out_dir, "frames"))
            
    camera_stream(out_dir)