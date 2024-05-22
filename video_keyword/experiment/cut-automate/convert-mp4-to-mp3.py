from moviepy.editor import VideoFileClip
import numpy as np
from tqdm import tqdm
from scipy.signal import correlate

def extract_audio(video_path):
    video = VideoFileClip(video_path)
    audio = video.audio
    newname = video_path.split('.')[0]
    print(newname)
    audio.write_audiofile(f'{newname}.mp3',codec='libmp3lame')
    # return audio.to_soundarray(fps=44100), audio.fps

if __name__ == "__main__":
    extract_audio('../dataset/infringement/IR-IN-VI-5.mp4')