from google.colab import files
import shutil
import os
import argparse

# 메인 프로그램이 실행될 때만 아래 코드를 실행하도록 설정 (다른 모듈로 import 될 때 실행되지 않도록 보호)
if __name__ == "__main__":
    # 명령줄 인자를 처리하기 위한 ArgumentParser 객체 생성
    parser = argparse.ArgumentParser()
    
    # "--type" 인자를 추가, 필수 입력 값으로 설정하고 설명을 추가
    parser.add_argument("--type", type=str, required=True, help="type of file to upload")
    
    # 명령줄 인자 파싱
    args = parser.parse_args()
    
    # 파싱된 "type" 인자의 값을 저장
    file_type = args.type

    # 현재 작업 디렉토리를 가져옴
    basepath = os.getcwd()
    
    # Google Colab에서 파일을 업로드하도록 요청, 업로드된 파일은 딕셔너리 형태로 반환됨
    uploaded = files.upload() # 업로드된 파일들을 가져옴
    
    # 파일 타입이 'zip', 'audio', 'video' 중 하나인지 확인
    assert(file_type in ['zip', 'audio', 'video'])
    
    # 파일 타입이 "zip"인 경우
    if file_type == "zip":
        # zip 파일을 저장할 경로 지정
        upload_path = "./custom_character_voice/"
        
        # 업로드된 각 파일에 대해 처리
        for filename in uploaded.keys():
            # 업로드된 파일을 지정된 위치로 이동하고 이름을 "custom_character_voice.zip"으로 변경
            shutil.move(os.path.join(basepath, filename), os.path.join(upload_path, "custom_character_voice.zip"))
    
    # 파일 타입이 "audio"인 경우
    elif file_type == "audio":
        # audio 파일을 저장할 경로 지정
        upload_path = "./raw_audio/"
        
        # 업로드된 각 파일에 대해 처리
        for filename in uploaded.keys():
            # 업로드된 파일을 지정된 위치로 이동, 파일명은 원본 그대로 유지
            shutil.move(os.path.join(basepath, filename), os.path.join(upload_path, filename))
    
    # 파일 타입이 "video"인 경우
    elif file_type == "video":
        # video 파일을 저장할 경로 지정
        upload_path = "./video_data/"
        
        # 업로드된 각 파일에 대해 처리
        for filename in uploaded.keys():
            # 업로드된 파일을 지정된 위치로 이동, 파일명은 원본 그대로 유지
            shutil.move(os.path.join(basepath, filename), os.path.join(upload_path, filename))
