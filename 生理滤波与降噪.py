import os
import re
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
from scipy.fftpack import fft
from scipy.stats import skew, kurtosis
from scipy import stats

def extract_features(signal):
    # 时域特征
    # 均值、标准差、最大值、最小值、峰峰值、均方根
    mean = np.mean(signal)
    std = np.std(signal)
    max_val = np.max(signal)
    min_val = np.min(signal)
    peak_to_peak = max_val - min_val
    rms = np.sqrt(np.mean(signal**2))

    # 频域特征
    # 频谱能量、频谱熵、频谱质心、频谱方差、频谱偏度、频谱峰度
    spectrum = np.abs(fft(signal))
    spectrum[spectrum == 0] = 1e-10  # 将零替换成一个极小的正数,以防divide 0 error
    freq_energy = np.sum(spectrum**2)
    freq_entropy = -np.sum(spectrum * np.log2(spectrum))
    freq_centroid = np.sum(np.arange(len(spectrum)) * spectrum) / np.sum(spectrum)
    freq_variance = np.sum(((np.arange(len(spectrum)) - freq_centroid) ** 2) * spectrum) / np.sum(spectrum)
    freq_skewness = skew(spectrum)
    freq_kurtosis = kurtosis(spectrum)

    return [mean, std, max_val, min_val, peak_to_peak, rms, freq_energy, freq_entropy, freq_centroid, freq_variance, freq_skewness, freq_kurtosis]
def read_file(file_path):
    content = pd.read_csv(file_path)
    return content

def outlier_remove(data):
    z_scores = stats.zscore(data)
    abs_z_scores = np.abs(z_scores)
    outlier_mask = (abs_z_scores < 3)
    # 将异常值替换为均值，保持数据长度不变
    data_mean = data.loc[~outlier_mask].mean()
    data.loc[outlier_mask] = data_mean
    return data
def butter_lowpass(cutoff, fs, order=6):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a
def butter_bandpass(lowcut, highcut, fs, order=6):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=6, btype='low'):
    if btype == 'low':
        b, a = butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
    elif btype == 'band':
        b, a = butter_bandpass(cutoff[0], cutoff[1], fs, order=order)
        y = lfilter(b, a, data)
    return y

def feature_generate(group_size, data, feature_list):
    for i in range(0, len(data), group_size):
        group_data = data[i:i + group_size]
        feature_list.append(extract_features(group_data))
    return feature_list

def timestamp_group(data, time_stamp_name,signal_name):
    # 时间戳转换为datetime格式
    data[time_stamp_name] = pd.to_datetime(data[time_stamp_name])
    # 计算时间戳之间的差异（单位:秒）
    data[f'time_diff_{signal_name}'] = (data[time_stamp_name] - data[time_stamp_name].shift()).dt.total_seconds()
    # 定义差异阈值
    threshold = 60
    data['segment'] = (data[f'time_diff_{signal_name}'] > threshold).cumsum()
    return data


if __name__ == '__main__':
    # gsr采样频率
    gsr_fs = 40.0
    # ppg采样频率
    ppg_fs = 20.0
    # acc采样频率
    acc_fs = 20.0
    # gsr截止频率
    gsr_cutoff = 1.0
    # ppg带通截止频率
    ppg_cutoff = [1.0, 8.0]
    # acc带通截止频率(排除低频的重力分量和高频的运动)
    acc_cutoff = [0.5, 5.0]
    # 滤波器阶数
    order = 6

    # 指定要搜索的目录
    directory = "E:/DAPPER/Processed/Processed"

    filenames = os.listdir(directory)
    numbers = set()
    for filename in filenames:
        match = re.match(r'Processed_(\d+)_', filename)
        if match:
            numbers.add(int(match.group(1)))

    for number in numbers:
        print("Processing file: ", number)
        gsr_content = read_file(f"{directory}/Processed_{number}_GSR.csv")
        ppg_content = read_file(f"{directory}/Processed_{number}_PPG.csv")
        acc_content = read_file(f"{directory}/Processed_{number}_ACC.csv")
        print(len(gsr_content), len(ppg_content), len(acc_content))

        # 连续时间分段处理
        gsr_content = timestamp_group(gsr_content, 'csv_time_GSR', 'GSR')
        ppg_content = timestamp_group(ppg_content, 'csv_time_PPG', 'PPG')
        acc_content = timestamp_group(acc_content, 'csv_time_motion', 'ACC')

        gsr_groups = gsr_content.groupby('segment')
        ppg_groups = ppg_content.groupby('segment')
        acc_groups = acc_content.groupby('segment')

        gsr_groups_list = []
        ppg_groups_list = []
        acc_groups_list = []

        for segment_index, gsr_group in gsr_groups:
            """处理GSR生理信号数据"""
            gsr_data_temp = pd.Series(gsr_group['GSR'])
            gsr_data = gsr_data_temp.copy()

            # 使用Z-score方法去除异常数据对GSR数据进行预处理
            gsr_data = outlier_remove(gsr_data)

            # 使用rolling方法对GSR数据进行滑动窗口处理
            rolling_mean = gsr_data.rolling(window=5).mean()
            # 使用均值对缺失值进行填充
            rolling_mean_avg = rolling_mean.mean()
            gsr_data = gsr_data.fillna(rolling_mean_avg)

            # 使用滤波器对GSR数据进行低通滤波
            gsr_btype = 'low'
            gsr_filtered_data = butter_lowpass_filter(gsr_data, gsr_cutoff, gsr_fs, order, gsr_btype)
            gsr_groups_list.append(gsr_filtered_data)
        # 数据合并
        gsr_filer_data = np.concatenate(gsr_groups_list)

        for segment_index, ppg_group in ppg_groups:
            """处理PPG生理信号数据"""
            ppg_data_temp = pd.Series(ppg_group['PPG'])
            ppg_data = ppg_data_temp.copy()

            ppg_data = outlier_remove(ppg_data)
            ppg_btype = 'band'
            # 使用滤波器对PPG数据进行带通滤波
            ppg_filtered_data = butter_lowpass_filter(ppg_data, ppg_cutoff, ppg_fs, order, ppg_btype)
            ppg_groups_list.append(ppg_filtered_data)
        ppg_filter_data = np.concatenate(ppg_groups_list)
        
        for segment_index, acc_group in acc_groups:
            index_list = ['Motion_dataX', 'Motion_dataY', 'Motion_dataZ']
            combined_data = np.empty((len(acc_group), 0))
            for index in index_list:
                acc_data_temp = acc_group[index]
                acc_data = acc_data_temp.copy()
                acc_data = outlier_remove(acc_data)
                # 使用滤波器对ACC数据进行带通滤波
                acc_filtered_data = butter_lowpass_filter(acc_data, acc_cutoff, acc_fs, order, ppg_btype)
                # 将滤波后的数据添加到combined_data上
                combined_data = np.concatenate((combined_data, acc_filtered_data.reshape(-1, 1)), axis=1)
            acc_groups_list.append(combined_data)
        acc_filter_data = np.concatenate(acc_groups_list)

        # # csv文件写入
        # new_gsr = pd.DataFrame(gsr_filtered_data, columns=['GSR'])
        # new_ppg = pd.DataFrame(ppg_filtered_data, columns=['PPG'])
        # new_acc = pd.DataFrame(combined_data, columns=['Motion_dataX', 'Motion_dataY', 'Motion_dataZ'])
        # new_gsr['csv_time_GSR'] = gsr_content['csv_time_GSR']
        # new_ppg['csv_time_PPG'] = ppg_content['csv_time_PPG']
        # new_acc['csv_time_motion'] = acc_content['csv_time_motion']
        #
        # new_gsr.to_csv(f"Processed_handle_{number}_GSR.csv", index=False)
        # new_ppg.to_csv(f"Processed_handle_{number}_PPG.csv", index=False)
        # new_acc.to_csv(f"Processed_handle_{number}_ACC.csv", index=False)
        break

        """处理时域特征与频域特征"""
        gsr_size = int(30 * gsr_fs)
        ppg_size = int(30 * ppg_fs)
        acc_size = int(30 * ecg_fs)

        gsr_features = []
        ppg_features = []
        acc_features = []

        gsr_features = feature_generate(gsr_size, gsr_filter_data, gsr_features)
        ppg_features = feature_generate(ppg_size, ppg_filter_data, ppg_features)

        acc_features_list = []
        for i in range(acc_filter_data.shape[1]):
            # 获取当前轴数据,分别是x,y,z
            axis_data = acc_filter_data[:, i]
            axis_features = []
            axis_features = feature_generate(acc_size, axis_data, axis_features)
            acc_features_list.append(axis_features)

        acc_features_list = np.array(acc_features_list)


