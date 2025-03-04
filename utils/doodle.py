# import numpy as np
# from pprint import pprint

# # read npy file

# def read_npy(file_path):
#     return np.load(file_path, allow_pickle=True)

# file = read_npy("/home/g202302610/Code/OpFlow-SLR/datasets/phoenix2014/train_info.npy").item()
# print(len(file))

# all = []
# for k in file.keys():
#     if k == "prefix": 
#         print(file[k])
#         continue

#     num_frames = file[k]["num_frames"]
#     all.append(num_frames)

# print(sum(all)/len(all))

# # print statistics on number of frames
# print("min length of frames: ", min(all))
# print("max length of frames: ", max(all))

# # quartiles
# print("15th percentile: ", np.percentile(all, 15))
# print("25th percentile: ", np.percentile(all, 25))
# print("50th percentile: ", np.percentile(all, 50))
# print("75th percentile: ", np.percentile(all, 75))
# print("75th percentile: ", np.percentile(all, 95))



# # max length of frames:  300


import cv2
import numpy as np
import subprocess
import os

def extract_iframe_mv_residual(video_path, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Extract I-Frame
    iframe_command = [
        'ffmpeg', '-i', video_path, '-vf', 'select=eq(pict_type\,I)', '-vsync', 'vfr',
        os.path.join(output_folder, 'iframe_%03d.png')
    ]
    subprocess.run(iframe_command)

    # Extract Motion Vectors (MV)
    mv_command = [
        'ffmpeg', '-flags2', '+export_mvs', '-i', video_path, '-vf', 'codecview=mv=pf+bf+bb',
        os.path.join(output_folder, 'mv_frame_%03d.png')
    ]
    subprocess.run(mv_command)

    # Extract Residual Frames
    residual_command = [
        'ffmpeg', '-i', video_path, '-vf', 'extractplanes=y',
        os.path.join(output_folder, 'residual_frame_%03d.png')
    ]
    subprocess.run(residual_command)

def save_hstacked_frames(output_folder, save_folder):
    # Ensure the save folder exists
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Load the extracted frames
    iframe_files = sorted([f for f in os.listdir(output_folder) if f.startswith('iframe')])
    mv_files = sorted([f for f in os.listdir(output_folder) if f.startswith('mv_frame')])
    residual_files = sorted([f for f in os.listdir(output_folder) if f.startswith('residual_frame')])

    for idx, (iframe_file, mv_file, residual_file) in enumerate(zip(iframe_files, mv_files, residual_files)):
        iframe = cv2.imread(os.path.join(output_folder, iframe_file))
        mv_frame = cv2.imread(os.path.join(output_folder, mv_file))
        residual_frame = cv2.imread(os.path.join(output_folder, residual_file))

        # Stack frames horizontally
        stacked_frames = np.hstack((iframe, mv_frame, residual_frame))

        # Save the stacked frames
        save_path = os.path.join(save_folder, f'stacked_frame_{idx:03d}.png')
        cv2.imwrite(save_path, stacked_frames)
        print(f'Saved: {save_path}')

if __name__ == "__main__":
    video_path = '/data/sharedData/CSL/color/000/P22_s1_00_2._color.mp4'  # Replace with your video path
    output_folder = '/home/g202302610/Code/OpFlow-SLR/test'
    save_folder = '/home/g202302610/Code/OpFlow-SLR/test'

    # Extract frames
    extract_iframe_mv_residual(video_path, output_folder)

    # Save horizontally stacked frames
    save_hstacked_frames(output_folder, save_folder)