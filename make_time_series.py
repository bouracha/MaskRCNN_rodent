import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.mstats import gmean
from shutil import copyfile
import os

import pickle
import umap


def eliminate_impossible_positions(df, variable):
    df[str(variable)+'_likelihood'][ (df[str(variable)+'_x'] < 0) | (df[str(variable)+'_x'] > 1200) ] = 0.0
    df[str(variable)+'_x'][ (df[str(variable)+'_x'] < 0) | (df[str(variable)+'_x'] > 1200) ] = np.nan
    df[str(variable)+'_likelihood'][ (df[str(variable)+'_y'] < 0) | (df[str(variable)+'_y'] > 700) ] = 0.0
    df[str(variable)+'_y'][ (df[str(variable)+'_y'] < 0) | (df[str(variable)+'_y'] > 700) ] = np.nan
    return df

def nanify_low_likelihoods(df, variable, threshold=0.5):
    df[str(variable)+'_x'][ df[str(variable)+'_likelihood'] < threshold ] = np.nan
    df[str(variable)+'_y'][ df[str(variable)+'_likelihood'] < threshold ] = np.nan
    return df


def format_input_csv(vid_num, video_path, video_file_names, batch, threshold=0.5, re_write_csv=False):
    "Assumes structure: leftear; rightear; nose; lefthand; righthand;"
    "                   leftfoot; rightfoot; tailbase; backcurve"
    "Returns: with sensible column titles and only values in columns"

    data = pd.read_csv(str(video_path) + str(vid_num) + str(video_file_names))

    for i in range(0, data.columns.shape[0]):
        data = data.rename(columns={str(data.columns[i]): str(data.iloc[0][i]) + '_' + str(data.iloc[1][i])})

    data = data.drop(data.index[0:2])
    data = data.rename(columns={str(data.columns[0]): 'frame_index'})
    data = data.set_index('frame_index')

    data = data.astype(float)

    data = eliminate_impossible_positions(data, 'leftear')
    data = eliminate_impossible_positions(data, 'rightear')
    data = eliminate_impossible_positions(data, 'nose')
    data = eliminate_impossible_positions(data, 'lefthand')
    data = eliminate_impossible_positions(data, 'righthand')

    if re_write_csv:
        ## Rewrite csv with above corrections of impossible values
        data.to_csv(batch + 'csv_formatted/' + str(vid_num) + str(video_file_names))

    # Set values to nans if likelihood is less than threshold
    data = nanify_low_likelihoods(data, 'leftear', threshold=threshold)
    data = nanify_low_likelihoods(data, 'rightear', threshold=threshold)
    data = nanify_low_likelihoods(data, 'nose', threshold=threshold)
    data = nanify_low_likelihoods(data, 'lefthand', threshold=threshold)
    data = nanify_low_likelihoods(data, 'righthand', threshold=threshold)

    if re_write_csv:
        ## Rewrite csv with above corrections of impossible values
        data.to_csv(batch + 'like_threshold' + str(int(10 * threshold)) + '/' + str(vid_num) + str(video_file_names))

    return data

def max_num_contiguous_nans(df, variable):
    "Returns: maximum number of contiguous nans for this variable"
    max_num_nans = df[str(variable)+'_x'].isnull().astype(int).groupby(df[str(variable)+'_x'].notnull().astype(int).cumsum()).sum().max()
    return max_num_nans


def mean_likelihoods(vid_num, metrics_dataframe, formatted_data, threshold=0.5):
    max_gap_leftear = max_num_contiguous_nans(formatted_data, 'leftear')
    max_gap_rightear = max_num_contiguous_nans(formatted_data, 'rightear')
    max_gap_nose = max_num_contiguous_nans(formatted_data, 'nose')
    max_gap_lefthand = max_num_contiguous_nans(formatted_data, 'lefthand')
    max_gap_righthand = max_num_contiguous_nans(formatted_data, 'righthand')
    max_gap = np.max([max_gap_leftear, max_gap_rightear, max_gap_nose, max_gap_lefthand, max_gap_righthand])

    metrics_dataframe = metrics_dataframe.append({'vid_num': vid_num,
                                                  'num_frames': formatted_data.shape[0],
                                                  'leftear': formatted_data['leftear_likelihood'].mean(),
                                                  'rightear': formatted_data['rightear_likelihood'].mean(),
                                                  'nose': formatted_data['nose_likelihood'].mean(),
                                                  'lefthand': formatted_data['lefthand_likelihood'].mean(),
                                                  'righthand': formatted_data['righthand_likelihood'].mean(),
                                                  'leftfoot': formatted_data['leftfoot_likelihood'].mean(),
                                                  'rightfoot': formatted_data['rightfoot_likelihood'].mean(),
                                                  'tailbase': formatted_data['tailbase_likelihood'].mean(),
                                                  'backcurve': formatted_data['backcurve_likelihood'].mean(),
                                                  'leftear_num_nans': formatted_data['leftear_x'][
                                                      formatted_data['leftear_likelihood'] < threshold].shape[0],
                                                  'rightear_num_nans': formatted_data['rightear_x'][
                                                      formatted_data['rightear_likelihood'] < threshold].shape[0],
                                                  'nose_num_nans': formatted_data['nose_x'][
                                                      formatted_data['nose_likelihood'] < threshold].shape[0],
                                                  'lefthand_num_nans': formatted_data['lefthand_x'][
                                                      formatted_data['lefthand_likelihood'] < threshold].shape[0],
                                                  'righthand_num_nans': formatted_data['righthand_x'][
                                                      formatted_data['righthand_likelihood'] < threshold].shape[0],
                                                  'max_gap': max_gap},
                                                 ignore_index=True)

    return metrics_dataframe


def dlc_to_metrics_dataframe(num_videos, dlc_path, batch, video_file_names):
    metrics_dataframe = pd.DataFrame(columns=['vid_num',
                                              'leftear', 'rightear', 'nose', 'lefthand', 'righthand',
                                              'leftfoot', 'rightfoot', 'tailbase', 'backcurve'])

    os.mkdir(batch)
    os.mkdir(batch + '/csv_formatted/')
    os.mkdir(batch + '/like_threshold5/')

    for vid_num in range(1, num_videos):
        print(str(vid_num) + "/" + str(num_videos - 1), end="\r")

        formatted_data = format_input_csv(vid_num, dlc_path, video_file_names, batch, threshold=0.5, re_write_csv=True)
        #Remove videos less than 60 frames
        if formatted_data.shape[0] < 60:
            continue

        metrics_dataframe = mean_likelihoods(vid_num, metrics_dataframe, formatted_data, threshold=0.1)

    metrics_dataframe = metrics_dataframe.set_index('vid_num')

    # Create new dataframe metrics
    metrics_dataframe['mean_all_markers'] = metrics_dataframe[
        ['leftear', 'rightear', 'nose', 'lefthand', 'righthand', 'leftfoot', 'rightfoot', 'tailbase',
         'backcurve']].mean(axis=1)
    metrics_dataframe['geomean_all_markers'] = gmean(metrics_dataframe[
                                                         ['leftear', 'rightear', 'nose', 'lefthand', 'righthand',
                                                          'leftfoot', 'rightfoot', 'tailbase', 'backcurve']], axis=1)
    metrics_dataframe['mean_head_hands'] = metrics_dataframe.iloc[:, 0:5].mean(axis=1)
    metrics_dataframe['geomean_head_hands'] = gmean(metrics_dataframe.iloc[:, 0:5], axis=1)
    metrics_dataframe['nan_ratio'] = metrics_dataframe[
                                         ['leftear_num_nans', 'rightear_num_nans', 'nose_num_nans', 'lefthand_num_nans',
                                          'righthand_num_nans']].mean(axis=1) / metrics_dataframe['num_frames']

    metrics_dataframe.to_csv(batch + 'metrics_dataframe5.csv')

    return metrics_dataframe


def select_videos_using_geomean(batch, geomean_threshold):
    metrics_dataframe = pd.read_csv(batch + 'metrics_dataframe5.csv')
    metrics_dataframe = metrics_dataframe.set_index('vid_num')

    df_reduced = metrics_dataframe.loc[(metrics_dataframe['geomean_head_hands'] > geomean_threshold)]
    list_of_indices = df_reduced.index.values.tolist()
    pd.DataFrame(list_of_indices).astype(int).to_csv(batch + '/best_list.csv')

    return list_of_indices


def remove_diffs_for_bodypart(data, bodypart):
    booles_diff_above_threshold = np.sqrt(
        (abs(data[bodypart + '_x'].diff())) ** 2 + (abs(data[bodypart + '_y'].diff())) ** 2) > 10.0
    data[bodypart + '_x'][booles_diff_above_threshold] = np.nan
    data[bodypart + '_y'][booles_diff_above_threshold] = np.nan

    return data


def remove_large_diffs(best_videos_list, batch):
    num_processed = 0

    os.mkdir(batch + '/preprocessed_diff/')

    for i in best_videos_list:
        print(str(num_processed) + "/" + str(len(best_videos_list)), end="\r")
        data = pd.read_csv(batch + 'like_threshold5/' + str(i) + 'DeepCut_resnet50_First_batchJun28shuffle1_900000.csv')

        data = remove_diffs_for_bodypart(data, 'leftear')
        data = remove_diffs_for_bodypart(data, 'rightear')
        data = remove_diffs_for_bodypart(data, 'nose')
        data = remove_diffs_for_bodypart(data, 'lefthand')
        data = remove_diffs_for_bodypart(data, 'righthand')

        data.to_csv(batch + 'preprocessed_diff/' + str(i) + '.csv')

        num_processed += 1

    return None


def interpolate_bodypart(data, bodypart, success):
    n_f = data[bodypart + '_x'].size
    # If first or last value is null set to average so we interpolate the entire range
    if data[bodypart + '_x'].isnull()[0] == True:
        if np.isnan(data[bodypart + '_x'].mean()):
            data[bodypart + '_x'][0] = 640.0
            data[bodypart + '_y'][0] = 350.0
        else:
            data[bodypart + '_x'][0] = data[bodypart + '_x'].mean()
            data[bodypart + '_y'][0] = data[bodypart + '_y'].mean()
    if data[bodypart + '_x'].isnull()[n_f - 1] == True:
        if np.isnan(data[bodypart + '_x'].mean()):
            data[bodypart + '_x'][n_f - 1] = 640.0
            data[bodypart + '_y'][n_f - 1] = 350.0
        else:
            data[bodypart + '_x'][n_f - 1] = data[bodypart + '_x'].mean()
            data[bodypart + '_y'][n_f - 1] = data[bodypart + '_y'].mean()

    try:
        data[bodypart + '_x'] = data[bodypart + '_x'].interpolate(method='polynomial', order=1)
        data[bodypart + '_y'] = data[bodypart + '_y'].interpolate(method='polynomial', order=1)
    except:
        # In the rare case that there are insufficient number of datapoints we won't use video
        success = False

    return data, success


def interpolate_files(best_videos_list, batch):
    num_processed = 0
    num_successful = 0

    os.mkdir(batch + '/interpolated/')

    for i in best_videos_list:
        print(str(num_processed) + "/" + str(len(best_videos_list)), end="\r")
        data = pd.read_csv('/home/pn/conv_autoencoder/'+str(batch)+'preprocessed_diff/' + str(i) + '.csv')

        success = True
        data, success = interpolate_bodypart(data, 'leftear', success)
        data, success = interpolate_bodypart(data, 'rightear', success)
        data, success = interpolate_bodypart(data, 'nose', success)
        data, success = interpolate_bodypart(data, 'lefthand', success)
        data, success = interpolate_bodypart(data, 'righthand', success)

        if success:
            data.to_csv(batch + 'interpolated/' + str(i) + '.csv')
            num_successful += 1
        num_processed += 1

    print(num_successful, "sucessfully interpolated out of", len(best_videos_list))
    return None


def get_all_windows(time_series_path, batch,  window_size=60, window_stride=1):
    videos_list = list(pd.read_csv('/home/pn/conv_autoencoder/'+str(batch)+'best_long_list.csv').iloc[:, 1])

    first_run = True
    for vid in videos_list:
        print(str(vid) + "/" + str(videos_list[-1:]), end="\r")
        if first_run:
            all_windows = time_series_to_window(time_series_path + str(vid) + '.csv', window_size, window_stride)
            # all_windows_images = video_to_window(video_path + str(vid) + '.avi', window_size=60, window_stride=1)
            first_run = False
            continue
        windows = time_series_to_window(time_series_path + str(vid) + '.csv', window_size, window_stride)
        # windows_images = video_to_window(video_path + str(vid) + '.csv', window_size=60, window_stride=1)
        all_windows = np.concatenate((all_windows, windows), axis=0)
        # all_windows_images = np.concatenate((all_windows_images, windows_images), axis=0)

    return all_windows  # , all_windows_images


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


if __name__ == '__main__':

    batch = 'batch2/'
    num_vids = 16033
    batch_path = '/home/pn/Desktop/deeplabcut/new_videos_7_M/'

    #metrics_dataframe = dlc_to_metrics_dataframe(num_vids, batch_path, batch, 'DeepCut_resnet50_First_batchJun28shuffle1_900000.csv')

    #best_videos_list = select_videos_using_geomean(batch, geomean_threshold=0.3)

    #remove_large_diffs(best_videos_list, batch)
    #interpolate_files(best_videos_list, batch)

    #best_videos_list = list(pd.read_csv('/home/pn/conv_autoencoder/' + str(batch) + 'best_long_list.csv').iloc[:, 1])

    reducer = pickle.load(open('/home/pn/conv_autoencoder/keras-autoencoders/umap_200_0.sav', 'rb'))
    window_size, window_stride = 60, 1

    all_windows = get_all_windows('/home/pn/conv_autoencoder/'+str(batch)+'interpolated_long/', batch, window_size, window_stride)

    data = np.reshape(all_windows, (len(all_windows), 600))

    embedding = reducer.transform(data)

    plt.figure(figsize=(20, 20))
    plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.002, s=3)
    plt.show()































