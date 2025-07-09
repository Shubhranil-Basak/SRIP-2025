import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.signal import savgol_filter
from scipy.signal import butter, filtfilt

fs = 1000

def highpass_filter(data, cutoff=20, fs=1000, order=4):
    data = np.asarray(data, dtype=float)  # Ensure data is numeric
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

def notch_filter(data, freq_a = 48, freq_b = 52, fs=1000, quality_factor=30):
    data = np.asarray(data, dtype=float)  # Ensure data is numeric
    nyquist = 0.5 * fs
    freq_normalized_a = freq_a / nyquist
    freq_normalized_b = freq_b / nyquist
    b, a = signal.butter(2, [freq_normalized_a, freq_normalized_b], btype='bandstop')
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

def bandpass_filter(data, low=20, high=450, fs=1000, order=4):
    data = np.asarray(data, dtype=float)  # Ensure data is numeric
    nyquist = 0.5 * fs
    low_normalized = low / nyquist
    high_normalized = high / nyquist
    b, a = signal.butter(order, [low_normalized, high_normalized], btype='band')
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

def lowpass_filter(data, cutoff=5, fs=1000, order=4):
    data = np.asarray(data, dtype=float)  # Ensure data is numeric
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    envelope = signal.filtfilt(b, a, data)
    return envelope

master_csv = pd.DataFrame({})

filename = 'hemanth'
part = 'Bicep'

for num in range(1, 6):
    num = str(num)
    print(num)
    data = pd.read_csv(f'./{filename}/{part}/txt/{filename+num}.txt')
    raw_emg3 = data['raw_emg'].values

    emg_highpass3 = highpass_filter(raw_emg3, cutoff=70, fs=fs)

    emg_bandpass3 = bandpass_filter(emg_highpass3, low=20, high=300, fs=fs)

    emg_notch3 = notch_filter(emg_bandpass3, fs=fs)

    data3_filtered = pd.DataFrame({
    'raw_emg': raw_emg3,
    'filtered_emg': emg_notch3
    })

    data3_filtered.to_csv(f'./{filename}/{part}/csv/{filename+num}.csv', index=False)

    # --- 1. Find positive maxima only ---
    positive_peaks, _ = signal.find_peaks(emg_notch3, distance=150)  # only positive peaks

    # --- 2. Select top 7 positive peaks by amplitude ---
    top_peaks = sorted(positive_peaks, key=lambda x: emg_notch3[x], reverse=True)[:1]
    top_peaks.sort()  # sort by position for segmentation

    # --- 3. Define boundaries as midpoints between peaks ---
    boundaries = [0]
    for i in range(len(top_peaks) - 1):
        midpoint = (top_peaks[i] + top_peaks[i + 1]) // 2
        boundaries.append(midpoint)
    boundaries.append(len(emg_notch3))  # end of signal

    # --- 4. Segment the signal based on boundaries ---
    segments = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        segments.append(data3_filtered.iloc[start:end].copy())

    window = 15  # Must be odd
    poly_order = 3
    emg_smoothed = savgol_filter(emg_notch3, window, poly_order, axis=0)

    accel = data[['raw_ax', 'raw_ay', 'raw_az']].values
    gyro = data[['raw_gx', 'raw_gy', 'raw_gz']].values

    def lowpass_filter(data, cutoff_freq, fs, order=4):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff_freq / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        filtered_data = filtfilt(b, a, data, axis=0)
        return filtered_data

    # Sample rate (Hz) and cutoff frequency (e.g., 5 Hz for slow movements)
    fs = 1000  # Adjust based on your sampling rate
    cutoff = 5
    accel_filtered = lowpass_filter(accel, cutoff, fs)
    gyro_filtered = lowpass_filter(gyro, cutoff, fs)

    window_size = 5  # Adjust based on noise level
    accel_smooth = pd.DataFrame(accel).rolling(window=window_size, center=True, min_periods=1).mean().values

    window_size = 5  # Adjust based on noise level
    gyro_smooth = pd.DataFrame(gyro).rolling(window=window_size, center=True, min_periods=1).mean().values

    gravity = np.mean(accel_filtered[:, 2])  # Assuming z-axis is vertical
    accel_dynamic = accel_filtered.copy()
    accel_dynamic[:, 2] -= gravity  # Subtract gravity from the vertical axis

    window = 15  # Must be odd
    poly_order = 3
    accel_smoothed = savgol_filter(accel_filtered, window, poly_order, axis=0)
    
    window = 15  # Must be odd
    poly_order = 3
    gyro_smoothed = savgol_filter(gyro_filtered, window, poly_order, axis=0)

    for i, seg in enumerate(segments):
        seg_copy = seg.copy()
        seg_copy.insert(0, 'Participant ID', 1)
        seg_copy.insert(1, 'File ID', int(num))
        seg_copy.insert(2, 'Segment ID', i)
        seg_copy.insert(3, 'Segment UID', f'{filename}_{num}_{i}')
        seg_copy.insert(4, 'Segment Start Time', seg.index[0])
        seg_copy.insert(5, 'Segment End Time', seg.index[-1])
        seg_copy.insert(8, 'accel_x', accel[seg.index, 0])
        seg_copy.insert(9, 'accel_y', accel[seg.index, 1])
        seg_copy.insert(10, 'accel_z', accel[seg.index, 2])
        seg_copy.insert(11, 'accel_x_filtered', accel_filtered[seg.index, 0])
        seg_copy.insert(12, 'accel_y_filtered', accel_filtered[seg.index, 1])
        seg_copy.insert(13, 'accel_z_filtered', accel_filtered[seg.index, 2])
        seg_copy.insert(14, 'accel_x_smooth', accel_smooth[seg.index, 0])
        seg_copy.insert(15, 'accel_y_smooth', accel_smooth[seg.index, 1])
        seg_copy.insert(16, 'accel_z_smooth', accel_smooth[seg.index, 2])
        seg_copy.insert(17, 'accel_x_SavGol', accel_smoothed[seg.index, 0])
        seg_copy.insert(18, 'accel_y_SavGol', accel_smoothed[seg.index, 1])
        seg_copy.insert(19, 'accel_z_SavGol', accel_smoothed[seg.index, 2])
        seg_copy.insert(20, 'gyro_x', gyro[seg.index, 0])
        seg_copy.insert(21, 'gyro_y', gyro[seg.index, 1])
        seg_copy.insert(22, 'gyro_z', gyro[seg.index, 2])
        seg_copy.insert(23, 'gyro_x_filtered', gyro_filtered[seg.index, 0])
        seg_copy.insert(24, 'gyro_y_filtered', gyro_filtered[seg.index, 1])
        seg_copy.insert(25, 'gyro_z_filtered', gyro_filtered[seg.index, 2])
        seg_copy.insert(26, 'gyro_x_smooth', gyro_smooth[seg.index, 0])
        seg_copy.insert(27, 'gyro_y_smooth', gyro_smooth[seg.index, 1])
        seg_copy.insert(28, 'gyro_z_smooth', gyro_smooth[seg.index, 2])
        seg_copy.insert(29, 'gyro_x_SavGol', gyro_smoothed[seg.index, 0])
        seg_copy.insert(30, 'gyro_y_SavGol', gyro_smoothed[seg.index, 1])
        seg_copy.insert(31, 'gyro_z_SavGol', gyro_smoothed[seg.index, 2])
        seg_copy.insert(32, 'part', part)
        master_csv = pd.concat([master_csv, seg_copy], ignore_index=True)

master_csv.to_csv('master_csv_Pronation_1.csv', index=False)
