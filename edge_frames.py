import cv2
import numpy as np
import scipy.misc
from matplotlib import pyplot as plt
import pandas as pd
import pickle
import umap


def frames_edges(vid_path, edge_threshold1=100, edge_threshold2=200):
    "numpy array images of dimensions (n_f, n_H, n_W)"
    "n_f = number of frames "
    "n_H = number of pixels vertically per frame"
    "n_W = number of pixels horizontally per frame"
    "n_C = number of channels per frame (RBG)"

    vidcap = cv2.VideoCapture(vid_path)
    n_f = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    success, image = vidcap.read()
    n_H, n_W, n_C = image.shape[0], image.shape[1], image.shape[2]
    images_edge = np.zeros([n_f, n_H, n_W])
    edge_image = cv2.Canny(image, edge_threshold1, edge_threshold2)

    count = 0
    images_edge[count, :, :] = edge_image
    success = True

    while success:
        count += 1
        success, image = vidcap.read()
        if success:
            edge_image = cv2.Canny(image, edge_threshold1, edge_threshold2)
            images_edge[count, :, :] = edge_image

    assert images_edge.shape == (n_f, n_H, n_W)

    return images_edge.astype(int), n_f, n_H, n_W, n_C


def time_series_to_window(file_path, window_size=60, window_stride=1):
    "Arguments: -Path to time series csv file"
    "           -Size of sliding window"
    "           -Stride of sliding windows"
    "Returns:   -Array of dimensions:"
    " n          num_windows, window_size, num_features"
    "ALSO MAKES SPATIALLY INDEPENDENT USING CENTROID"

    data = pd.read_csv(file_path)
    n_rows, _ = data.shape

    features = ['leftear_x', 'leftear_y', 'rightear_x', 'rightear_y', 'nose_x', 'nose_y', 'lefthand_x', 'lefthand_y',
                'righthand_x', 'righthand_y', ]
    #
    x_features = ['leftear_x', 'rightear_x', 'nose_x', 'lefthand_x', 'righthand_x']
    y_features = ['leftear_y', 'rightear_y', 'nose_y', 'lefthand_y', 'righthand_y']

    x_data = data[x_features]
    data['leftear_x'] = x_data['leftear_x'] - x_data.mean(axis=1)
    data['rightear_x'] = x_data['rightear_x'] - x_data.mean(axis=1)
    data['nose_x'] = x_data['nose_x'] - x_data.mean(axis=1)
    data['lefthand_x'] = x_data['lefthand_x'] - x_data.mean(axis=1)
    data['righthand_x'] = x_data['righthand_x'] - x_data.mean(axis=1)

    y_data = data[y_features]
    data['leftear_y'] = y_data['leftear_y'] - y_data.mean(axis=1)
    data['rightear_y'] = y_data['rightear_y'] - y_data.mean(axis=1)
    data['nose_y'] = y_data['nose_y'] - y_data.mean(axis=1)
    data['lefthand_y'] = y_data['lefthand_y'] - y_data.mean(axis=1)
    data['righthand_y'] = y_data['righthand_y'] - y_data.mean(axis=1)

    stack = np.stack(
        data[features].iloc[i:i + window_size] for i in range(0, n_rows - (window_size - 1), window_stride))

    return stack

def average_vid_from_range(reducer, x_min, x_max, y_min, y_max):
    mean_vids = np.zeros((60, 720, 1280))
    videos_list = list(pd.read_csv('/home/pn/Desktop/post_deeplabcut/batch1/best_list.csv').iloc[:, 1])
    num_videos_in_region = 0

    for vid in videos_list:
        print(str(vid) + "/" + str(videos_list[-1:]), end="\r")

        video_data = time_series_to_window('/home/pn/Desktop/post_deeplabcut/batch1/interpolated/' + str(vid) + '.csv',
                                           window_size, window_stride)

        video_data = np.reshape(video_data, (len(video_data), 600))
        video_embedding = reducer.transform(video_data)

        box_set = (video_embedding[:, 0] >= x_min) & (video_embedding[:, 0] <= x_max) & (
                    video_embedding[:, 1] >= y_min) & (video_embedding[:, 1] <= y_max)

        num_videos_in_region  += np.sum(box_set)

        if not any(box_set):
            continue

        try:
            frames_edge, n_f, n_H, n_W, n_C = frames_edges(
                '/home/pn/Desktop/deeplabcut/data_time_series/best_videos/' + str(vid) + '.avi', 32, 32)
        except:
            print('Video of size ' + str(n_f) + ' is too large..')
            continue

        frame_list = []
        for i in range(0, window_size):
            try:
                frame_list.append(frames_edge[i:i + len(box_set)][box_set].mean(axis=0, dtype=int))
            except:
                print('Video of size ' + str(n_f) + ' is too large..')
                continue
        mean_vid = np.stack(frame_list)
        mean_vids += mean_vid

    print('Region ' + str(x_min) + '_' + str(x_max) + '_' + str(y_min) + '_' + str(y_max) + ' has ' + str(num_videos_in_region) + ' videos')

    return mean_vids


def background_subtraction(x_min, x_max, y_min, y_max):

    cap = cv2.VideoCapture(str(x_min)+'_'+str(x_max)+'_'+str(y_min)+'_'+str(y_max)+'.avi')
    fgbg = cv2.createBackgroundSubtractorMOG2()
    out = cv2.VideoWriter(str(x_min)+'_'+str(x_max)+'_'+str(y_min)+'_'+str(y_max)+'_subtracted.avi', cv2.VideoWriter_fourcc(*'PIM1'), 25, (1280, 720))
    while(1):
        success, frame = cap.read()
        if success == False:
            break

        fgmask = fgbg.apply(frame)
        out_frame = cv2.cvtColor(np.uint8(fgmask), cv2.COLOR_GRAY2RGB)

        out.write(np.uint8(out_frame))


if __name__ == '__main__':

    window_size, window_stride = 60, 1
    reducer = pickle.load(open('umap_200_0.sav', 'rb'))

    #all_windows = get_all_windows('/home/pn/Desktop/post_deeplabcut/batch1/interpolated/', window_size, window_stride)

    corners = ((7.95,8.1,-2.85,-2.775), (8.5, 9.5, 0.5, 1.5), (8.9, 9.0, 0.9, 1.0), (8.7, 9.7, 0.5, 1.5), (-15, -10, -5, 0), (5, 10, -16, -12), (4, 8, -9, -6), (8, 10, 8, 10))
    for x_min, x_max, y_min, y_max in corners:

        mean_vids = average_vid_from_range(reducer, x_min, x_max, y_min, y_max)

        video = []
        for i in range(0, len(mean_vids)):
            video.append(cv2.cvtColor(np.uint8(mean_vids[i] * 255 / mean_vids.max()), cv2.COLOR_GRAY2RGB))
        video = np.stack(video)

        out = cv2.VideoWriter(str(x_min)+'_'+str(x_max)+'_'+str(y_min)+'_'+str(y_max)+'.avi', cv2.VideoWriter_fourcc(*'PIM1'), 25, (1280, 720))
        for i in range(len(mean_vids)):
            out.write(np.uint8(video[i]))

        background_subtraction(x_min, x_max, y_min, y_max)




























