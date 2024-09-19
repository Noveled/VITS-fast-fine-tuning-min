import os
from concurrent.futures import ThreadPoolExecutor

from moviepy.editor import AudioFileClip

# 비디오 파일들이 저장된 디렉토리 경로
video_dir = "./video_data/"

# 오디오 파일을 저장할 디렉토리 경로
audio_dir = "./raw_audio/"

# video_dir 내의 파일 목록을 가져옴
filelist = list(os.walk(video_dir))[0][2]

# 비디오 파일 목록을 생성하는 함수
def generate_infos():
    videos = []
    for file in filelist:
        # 파일 확장자가 ".mp4"인 파일들만 리스트에 추가
        if file.endswith(".mp4"):
            videos.append(file)
    return videos

# 개별 비디오 파일에서 오디오 클립을 추출하는 함수
def clip_file(file):
    # 비디오 파일에서 오디오 클립을 추출
    my_audio_clip = AudioFileClip(video_dir + file)

    # 추출한 오디오 클립을 지정된 디렉토리에 ".wav" 파일로 저장
    # 원본 비디오 파일명에서 "mp4"를 제거하고 "wav" 확장자로 저장
    my_audio_clip.write_audiofile(audio_dir + file.rstrip("mp4") + "wav")

# 프로그램의 메인 부분
if __name__ == "__main__":
    # mp4 파일 목록을 생성
    infos = generate_infos()

    # CPU 코어 수에 맞게 스레드를 생성하여 병렬로 작업을 처리
    # 최대 워커(worker)의 수는 현재 시스템의 CPU 코어 수로 지정
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        # clip_file 함수를 병렬로 실행하여 각 비디오 파일의 오디오를 추출
        executor.map(clip_file, infos)
