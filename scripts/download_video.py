import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor
from google.colab import files

# 현재 작업 디렉토리를 가져옴
basepath = os.getcwd()

# Google Colab에서 파일 업로드 기능을 사용하여 파일을 업로드, 업로드된 파일은 딕셔너리 형태로 반환됨
uploaded = files.upload()  # 파일 업로드

# 업로드된 각 파일에 대해 처리
for filename in uploaded.keys():
    # 업로드된 파일의 확장자가 ".txt"인지 확인 (아니면 AssertionError 발생)
    assert (filename.endswith(".txt")), "speaker-videolink info could only be .txt file!"
    
    # 업로드된 파일을 현재 작업 디렉토리에서 "./speaker_links.txt" 파일로 이동 및 이름 변경
    shutil.move(os.path.join(basepath, filename), os.path.join("./speaker_links.txt"))


# speaker_links.txt 파일에서 데이터를 읽어와 정보를 생성하는 함수
def generate_infos():
    infos = []
    
    # 'speaker_links.txt' 파일을 읽기 모드로 엶, 인코딩은 UTF-8로 설정
    with open("./speaker_links.txt", 'r', encoding='utf-8') as f:
        # 파일에서 모든 줄을 읽어옴
        lines = f.readlines()
    
    # 각 줄에 대해 처리
    for line in lines:
        # 줄 끝의 개행 문자와 공백을 제거
        line = line.replace("\n", "").replace(" ", "")
        
        # 빈 줄인 경우에는 처리하지 않고 건너뜀
        if line == "":
            continue
        
        # 각 줄을 '|' 문자로 구분하여 'speaker'와 'link'로 분리
        speaker, link = line.split("|")
        
        # speaker 이름과 무작위 숫자를 조합하여 고유한 파일 이름 생성
        filename = speaker + "_" + str(random.randint(0, 1000000))
        
        # 'link'와 'filename' 정보를 딕셔너리로 만들어 리스트에 추가
        infos.append({"link": link, "filename": filename})
    
    # 생성된 리스트를 반환
    return infos


# 동영상을 다운로드하는 함수
def download_video(info):
    # 'info' 딕셔너리에서 'link'와 'filename'을 가져옴
    link = info["link"]
    filename = info["filename"]
    
    # youtube-dl 명령어를 사용해 주어진 링크에서 동영상을 다운로드, 출력 파일 이름은 'filename'.mp4
    # --no-check-certificate 옵션은 SSL 인증서를 확인하지 않고 다운로드하도록 설정
    os.system(f"youtube-dl -f 0 {link} -o ./video_data/{filename}.mp4 --no-check-certificate")


# 메인 프로그램 실행
if __name__ == "__main__":
    # speaker_links.txt 파일을 읽고 다운로드할 비디오 정보 리스트 생성
    infos = generate_infos()
    
    # CPU 코어 수에 맞춰 최대 워커 스레드를 생성하여 동영상 다운로드 작업을 병렬로 실행
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        # 각 'info' 딕셔너리를 download_video 함수에 전달하여 병렬로 다운로드 실행
        executor.map(download_video, infos)
