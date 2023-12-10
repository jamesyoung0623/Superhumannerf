import cv2
from tqdm import tqdm
import glob

#filenames = sorted(glob.glob('../experiments/superhumannerf/zju_mocap/386/baseline/iter_106600/movement/*'))
filenames = sorted(glob.glob('../experiments/superhumannerf/wild/pitching/baseline/iter_106600/movement/*'))
frames = len(filenames)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None

for i in tqdm(range(frames)):
    filename = filenames[i]
    frame = cv2.imread(filename)

    if out is None:
        out = cv2.VideoWriter('output.mp4', fourcc, 10.0, (frame.shape[1], frame.shape[0]))

    out.write(frame)

# Release everything if job is finished
out.release()