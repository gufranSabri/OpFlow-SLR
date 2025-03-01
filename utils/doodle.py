import numpy as np
from pprint import pprint

# read npy file

def read_npy(file_path):
    return np.load(file_path, allow_pickle=True)

file = read_npy("/home/g202302610/Code/OpFlow-SLR/datasets/phoenix2014/train_info.npy").item()
print(len(file))

mi, ma = 1e9, -1
for k in file.keys():
    if k == "prefix": 
        print(file[k])
        continue
        # exit()
    num_frames = file[k]["num_frames"]
    mi = min(mi, num_frames)
    ma = max(ma, num_frames)
    print(file[k])

print(mi, ma)


# max length of frames:  300