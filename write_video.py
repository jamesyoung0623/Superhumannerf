import cv2
from tqdm import tqdm
import glob
import numpy as np

def write_movement():
    filenames = sorted(glob.glob('../experiments/superhumannerf/zju_mocap/394/baseline/latest/movement/*'))
    frames = len(filenames)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    for i in tqdm(range(frames)):
        frame = cv2.imread(filenames[i])

        if out is None:
            out = cv2.VideoWriter('output.mp4', fourcc, 10.0, (frame.shape[1], frame.shape[0]))

        out.write(frame)

    out.release()

def write_freeview():
    filenames_list = []
    for subject in ['377', '386', '387', '392', '393', '394']:
        filenames = sorted(glob.glob('../experiments/superhumannerf/zju_mocap/{}/baseline/latest/freeview_0/*'.format(subject)))
        filenames_list.append(filenames)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    for i in tqdm(range(len(filenames_list[0]))):
        upper_frame = None
        for filenames in filenames_list[:3]:
            frame = cv2.imread(filenames[i])
            if upper_frame is None:
                upper_frame = frame
            else:
                upper_frame = np.concatenate((upper_frame, frame), axis=1)

        lower_frame = None
        for filenames in filenames_list[3:]:
            frame = cv2.imread(filenames[i])
            if lower_frame is None:
                lower_frame = frame
            else:
                lower_frame = np.concatenate((lower_frame, frame), axis=1)

        whole_frame = np.concatenate((upper_frame, lower_frame), axis=0) 

        if out is None:
            out = cv2.VideoWriter('output.mp4', fourcc, 10.0, (whole_frame.shape[1], whole_frame.shape[0]))

        out.write(whole_frame)

    out.release()

def write_tpose():
    filenames_list = []
    for subject in ['377', '386', '387', '392', '393', '394']:
        filenames = sorted(glob.glob('../experiments/superhumannerf/zju_mocap/{}/baseline/latest/tpose/*'.format(subject)))
        filenames_list.append(filenames)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    for i in tqdm(range(len(filenames_list[0]))):
        upper_frame = None
        for filenames in filenames_list[:3]:
            frame = cv2.imread(filenames[i])
            if upper_frame is None:
                upper_frame = frame
            else:
                upper_frame = np.concatenate((upper_frame, frame), axis=1)

        lower_frame = None
        for filenames in filenames_list[3:]:
            frame = cv2.imread(filenames[i])
            if lower_frame is None:
                lower_frame = frame
            else:
                lower_frame = np.concatenate((lower_frame, frame), axis=1)

        whole_frame = np.concatenate((upper_frame, lower_frame), axis=0) 

        if out is None:
            out = cv2.VideoWriter('output.mp4', fourcc, 10.0, (whole_frame.shape[1], whole_frame.shape[0]))

        out.write(whole_frame)

    out.release()

# write_movement()
# write_freeview()
write_tpose()