import cv2
import numpy as np
import os
import glob
import re
import shutil
from tqdm import tqdm

MOTION_THRESHOLD = 75  # Adjust as needed to filter noise

def compute_optical_flow(prev_frame, next_frame):
    """Compute optical flow using Farneback method and apply a threshold to highlight significant movement."""
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Normalize to [0, 255]
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    magnitude = magnitude.astype(np.uint8)

    # Apply threshold: set low-motion areas to black (0)
    thresholded = np.where(magnitude > MOTION_THRESHOLD, magnitude, 0).astype(np.uint8)

    return thresholded


def numerical_sort_key(filename):
    match = re.search(r'fn(\d+)', filename)
    return int(match.group(1)) if match else float('inf') 


parent_dir = "/data/sharedData/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-256x256px/test"

dirs = os.listdir(parent_dir)
for dir in tqdm(dirs):
    frame_dirs = os.listdir(os.path.join(parent_dir, dir))

    for frame_dir in frame_dirs:
        # if "_of" in frame_dir:
        #     print(os.path.join(parent_dir, dir, frame_dir))
        #     shutil.rmtree(os.path.join(parent_dir, dir, frame_dir))
        #     continue
            
        # else:
        #     continue


        frame_paths = sorted(os.listdir(os.path.join(parent_dir, dir, frame_dir)))
        # remove non images
        frame_paths = [path for path in frame_paths if path.endswith(".png") or path.endswith(".jpg") or path.endswith(".jpeg")] 


        # Sort based on numbered filenames
        # frame_paths.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))
        if os.path.exists(os.path.join(parent_dir, dir, frame_dir + "_of")):
            shutil.rmtree(os.path.join(parent_dir, dir, frame_dir + "_of"))

        os.mkdir(os.path.join(parent_dir, dir, frame_dir + "_of"))
        frame_paths = sorted(frame_paths, key=numerical_sort_key)
        # print(frame_paths)

        frames = [cv2.imread(os.path.join(parent_dir, dir, frame_dir, path)) for path in frame_paths]

        # Create output folder to save images
        output_folder = os.path.join(parent_dir, dir, frame_dir + "_of")
        os.makedirs(output_folder, exist_ok=True)

        for i in range(1, len(frames)):
            prev_frame = frames[i-1]
            curr_frame = frames[i]
            
            heatmap_prev = compute_optical_flow(prev_frame, curr_frame)

            if i >= 4:
                frame_4_back = frames[i-4]
                heatmap_4_back = compute_optical_flow(frame_4_back, curr_frame)
            else:
                heatmap_4_back = np.zeros_like(heatmap_prev)

            if i >= 8:
                frame_8_back = frames[i-8]
                heatmap_8_back = compute_optical_flow(frame_8_back, curr_frame)
            else:
                heatmap_8_back = np.zeros_like(heatmap_prev)

            # Stack along channel dimension (convert to 3-channel grayscale)
            stacked_channel = np.dstack((heatmap_prev, heatmap_4_back, heatmap_8_back))

            # Save the image
            filename = os.path.join(output_folder, frame_paths[i])
            tqdm.write(f"Saving: {filename}")
            cv2.imwrite(filename, stacked_channel)

        tqdm.write(f"Saved optical flow images in: {output_folder}")
        tqdm.write("\n\n")
