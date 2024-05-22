import librosa
import numpy as np
from scipy.signal import find_peaks
from tqdm import tqdm

def load_audio(file_path):
    # 오디오 파일 로드 및 정규화
    signal, sr = librosa.load(file_path, sr=None)
    return signal / np.max(np.abs(signal)), sr

def cross_correlation(signal1, signal2):
    # 두 신호 간의 크로스 코릴레이션 계산
    return np.correlate(signal1, signal2, mode='full')

@profile
def find_correlations(a_signal, b_signal, sr, window_size):
    # A영상의 각 윈도우에 대해 B영상의 전체 오디오와 크로스 코릴레이션 계산
    correlations = []
    for start in tqdm(range(0, len(a_signal), window_size * sr)):
        end = start + window_size * sr
        window = a_signal[start:end]
        print(start)
        if len(window) == window_size * sr:
            correlation = cross_correlation(window, b_signal)
            # 피크 찾기
            peaks, _ = find_peaks(correlation)
            if peaks.size > 0:
                peak = peaks[np.argmax(correlation[peaks])]  # 가장 높은 피크
                peak_time = len(b_signal)-peak
                correlations.append((start / sr, end / sr, peak_time / sr))
    return correlations

# 오디오 파일 로드
a_signal, sr_a = load_audio('../IR-TI-VI-1.mp3')
b_signal, sr_b = load_audio('../OR-YO-VI-1.mp3')

# 샘플 레이트가 다르면 오류 처리
if sr_a != sr_b:
    raise ValueError("Sample rates of the two audio files must be the same.")

# 윈도우 크기 설정 (예: 1초)
window_size = 1  # seconds

# 크로스 코릴레이션 계산
correlations = find_correlations(a_signal, b_signal, sr_a, window_size)

# 결과 출력
for start_time, end_time, peak_time in correlations:
    print(f"Window from {start_time} to {end_time} seconds in A is similar to the section at {peak_time} seconds in B.")