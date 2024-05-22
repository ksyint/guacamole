import librosa
import soundfile as sf

def cut_audio(file_path, start_sec, end_sec, output_file):
    # 오디오 파일 로드
    y, sr = librosa.load(file_path, sr=None)
    
    # 초단위로 시간 인덱스 계산
    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)
    
    # 오디오 자르기
    y_cut = y[start_sample:end_sample]

    # 잘라낸 오디오를 새 파일로 저장
    sf.write(output_file, y_cut, sr)

if __name__ == '__main__':
# 예제 사용
    cut_audio('IR-IN-VI-5.mp3', 0, 4, 'cut0to4IR-IN-VI-5.mp3')