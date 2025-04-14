import numpy as np
from scipy import signal

def normalize_signal(data):
    """신호 정규화"""
    return (data - np.mean(data)) / (np.std(data) + 1e-6)

def remove_baseline(data, window_size=250):
    """베이스라인 제거"""
    baseline = signal.medfilt(data, window_size)
    return data - baseline

def apply_bandpass_filter(data, fs, lowcut=0.5, highcut=40.0):
    """대역 통과 필터 적용"""
    nyquist = fs / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    return signal.filtfilt(b, a, data)

def detect_r_peaks(data, fs, window_size=300):
    """R-peak 검출"""
    # 데이터 전처리
    filtered = remove_baseline(data)
    filtered = apply_bandpass_filter(filtered, fs)
    filtered = normalize_signal(filtered)
    
    # 피크 검출
    peaks = []
    min_peak_distance = int(0.2 * fs)  # 최소 200ms 간격
    
    for i in range(0, len(filtered) - window_size, window_size // 2):
        window = filtered[i:i + window_size]
        peak_idx = i + np.argmax(window)
        
        if not peaks or (peak_idx - peaks[-1]) >= min_peak_distance:
            peaks.append(peak_idx)
            
    return np.array(peaks)

def extract_segments(data, peaks, segment_size=300):
    """R-peak 주변 세그먼트 추출"""
    segments = []
    half_size = segment_size // 2
    
    for peak in peaks:
        start_idx = max(0, peak - half_size)
        end_idx = min(len(data), peak + half_size)
        
        if end_idx - start_idx == segment_size:
            segment = data[start_idx:end_idx]
            segments.append(segment)
            
    return np.array(segments) 