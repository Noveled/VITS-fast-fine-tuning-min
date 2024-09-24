# # pydub 및 ffmpeg 설치:
# pip install pydub
# sudo apt-get install ffmpeg

# # .mp3 파일로 변환하는 코드
from pydub import AudioSegment
import numpy as np

# 함수에서 반환된 값
sampling_rate, audio_data = hps.data.sampling_rate, audio

# numpy 배열을 pydub의 AudioSegment로 변환
audio_segment = AudioSegment(
    audio_data.tobytes(),  # 오디오 데이터를 바이트 형식으로 변환
    frame_rate=sampling_rate,
    sample_width=audio_data.dtype.itemsize,  # 샘플 폭 (바이트)
    channels=1  # 모노 오디오인 경우
)

# 오디오를 .mp3 파일로 저장
audio_segment.export("output_audio.mp3", format="mp3")
