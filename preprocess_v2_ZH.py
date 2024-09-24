import os
import argparse
import json
import sys
sys.setrecursionlimit(500000)  # Fix the error message of RecursionError: maximum recursion depth exceeded while calling a Python object.  You can change the number as you want.
# 파이썬의 재귀 한도를 늘려 RecursionError 방지. 이 숫자는 필요에 따라 조정 가능.

if __name__ == "__main__":
    # 명령줄 인자 처리기 생성
    parser = argparse.ArgumentParser()
    parser.add_argument("--add_auxiliary_data", type=bool, help="Whether to add extra data as fine-tuning helper") # 세부 조정에 도움이 되는 추가 데이터를 추가할지 여부
    parser.add_argument("--languages", default="CJE") # 사용할 언어 설정 (CJE: 중국어, 일본어, 영어)
    args = parser.parse_args()

    # 언어 설정에 따른 사용 언어 목록 정의
    if args.languages == "CJE":
        langs = ["[ZH]", "[JA]", "[EN]"] # 중국어, 일본어, 영어
    elif args.languages == "CJ":
        langs = ["[ZH]", "[JA]"] # 중국어, 일본어
    elif args.languages == "C":
        langs = ["[ZH]"] # 중국어


    new_annos = [] # 새로 병합할 주석 리스트


    # Source 1: transcribed short audios
    # 첫 번째 소스: transcribed short audios (짧은 오디오 파일에서 추출된 주석 데이터)
    if os.path.exists("short_character_anno.txt"): # 해당 파일이 존재할 경우
        with open("short_character_anno.txt", 'r', encoding='utf-8') as f:
            short_character_anno = f.readlines() # 모든 줄을 읽어옴
            new_annos += short_character_anno # 새로운 주석 리스트에 추가
    
    # Source 2: transcribed long audio segments
    # 두 번째 소스: transcribed long audio segments (긴 오디오 파일에서 추출된 주석 데이터)
    if os.path.exists("./long_character_anno.txt"): # 해당 파일이 존재할 경우
        with open("./long_character_anno.txt", 'r', encoding='utf-8') as f:
            long_character_anno = f.readlines() # 모든 줄을 읽어옴
            new_annos += long_character_anno # 새로운 주석 리스트에 추가

    # Get all speaker names
    # 모든 스피커 이름을 추출
    speakers = []
    for line in new_annos:
        path, speaker, text = line.split("|") # 주석 데이터에서 스피커 이름 추출
        if speaker not in speakers: # 스피커가 리스트에 없으면 추가
            speakers.append(speaker)

    # 주석 파일에 스피커가 없으면 에러 발생
    assert (len(speakers) != 0), "No audio file found. Please check your uploaded file structure." # 오디오 파일을 찾을 수 없습니다. 업로드한 파일 구조를 확인하세요.

    

    # 아래는 'sampled_audio4ft.txt' 내부의 데이터 형식
    # ./sampled_audio4ft/0.wav|specialweek|[JA]そうなんですけど、他のウマ娘さんを見るのは初めてでっ！[JA]
    # ./sampled_audio4ft/1.wav|specialweek|[JA]私、今日からトレセン学園に転入してきました、 スペシャルウィークって言います！！[JA]
    # ./sampled_audio4ft/2.wav|specialweek|[JA]うああぁ～～本物のウマ娘さん……！[JA]
    # ./sampled_audio4ft/474.wav|zhongli|[ZH]正是。[ZH]
    # ./sampled_audio4ft/475.wav|zhongli|[ZH]要是见到他，不妨也道一声节日快乐。[ZH]
    # ./sampled_audio4ft/476.wav|zhongli|[ZH]可惜…[ZH]
    # ./sampled_audio4ft/850.wav|vctk|[EN]"They had to explain it."[EN]
    # ./sampled_audio4ft/851.wav|vctk|[EN]"However, he will now have to wait for his chance to impress."[EN]
    # ./sampled_audio4ft/852.wav|vctk|[EN]"But the law is very clear on this."[EN]

    # Source 3 (Optional): sampled audios as extra training helpers
    # 세 번째 소스 (옵션): 추가적인 학습 보조 데이터를 사용하는 경우
    if args.add_auxiliary_data:
        with open("./sampled_audio4ft.txt", 'r', encoding='utf-8') as f:
            old_annos = f.readlines() # 추가 학습 데이터를 읽어옴
            
        # filter old_annos according to supported languages
        # 지원하는 언어에 맞게 필터링
        filtered_old_annos = []
        for line in old_annos:
            for lang in langs:
                if lang in line: # 지원 언어에 맞는 데이터만 필터링
                    filtered_old_annos.append(line)
        old_annos = filtered_old_annos

        # 필터링한 주석에서 스피커 정보 추출
        for line in old_annos:
            path, speaker, text = line.split("|")
            if speaker not in speakers:
                speakers.append(speaker)

        # 새로운 주석과 기존 주석의 데이터 수 계산
        num_old_voices = len(old_annos)
        num_new_voices = len(new_annos)

        # STEP 1: balance number of new & old voices
        # STEP 1: 새로운 주석과 기존 주석의 균형을 맞춤
        cc_duplicate = num_old_voices // num_new_voices
        if cc_duplicate == 0:
            cc_duplicate = 1 # 최소 한 번은 새로운 데이터를 복제


        # STEP 2: modify config file
        # STEP 2: 설정 파일 수정
        with open("./configs/finetune_speaker.json", 'r', encoding='utf-8') as f:
            hps = json.load(f) # 기존 설정 파일 로드

        # assign ids to new speakers
        # 새 스피커에 고유 ID 할당
        speaker2id = {}
        for i, speaker in enumerate(speakers):
            speaker2id[speaker] = i

        # modify n_speakers
        # 설정 파일의 스피커 수 수정
        hps['data']["n_speakers"] = len(speakers)

        # overwrite speaker names
        # 스피커 이름을 ID로 대체
        hps['speakers'] = speaker2id

        # 학습 관련 설정 수정 (로그 출력 주기, 평가 주기, 배치 크기 등)
        hps['train']['log_interval'] = 10
        hps['train']['eval_interval'] = 100
        hps['train']['batch_size'] = 16
        hps['data']['training_files'] = "final_annotation_train.txt"
        hps['data']['validation_files'] = "final_annotation_val.txt"

        # save modified config
        # 수정된 설정 파일 저장
        with open("./configs/modified_finetune_speaker.json", 'w', encoding='utf-8') as f:
            json.dump(hps, f, indent=2)

        # STEP 3: clean annotations, replace speaker names with assigned speaker IDs
        # STEP 3: 주석 데이터 정리 및 스피커 이름을 ID로 변환
        import text # 텍스트 클리닝을 위한 모듈

        cleaned_new_annos = []
        for i, line in enumerate(new_annos):
            path, speaker, txt = line.split("|")
            if len(txt) > 150: # 텍스트가 너무 긴 경우 건너뜀
                continue
            cleaned_text = text._clean_text(txt, hps['data']['text_cleaners']) # 텍스트 클리닝 수행
            cleaned_text += "\n" if not cleaned_text.endswith("\n") else ""
            cleaned_new_annos.append(path + "|" + str(speaker2id[speaker]) + "|" + cleaned_text)

        cleaned_old_annos = []
        for i, line in enumerate(old_annos):
            path, speaker, txt = line.split("|")
            if len(txt) > 150: # 텍스트가 너무 긴 경우 건너뜀
                continue
            cleaned_text = text._clean_text(txt, hps['data']['text_cleaners']) # 텍스트 클리닝 수행
            cleaned_text += "\n" if not cleaned_text.endswith("\n") else ""
            cleaned_old_annos.append(path + "|" + str(speaker2id[speaker]) + "|" + cleaned_text)

        # merge with old annotation
        # 기존 주석과 새로운 주석을 병합
        final_annos = cleaned_old_annos + cc_duplicate * cleaned_new_annos

        # save annotation file
        # 학습용 주석 파일 저장
        with open("./final_annotation_train.txt", 'w', encoding='utf-8') as f:
            for line in final_annos:
                f.write(line)

        # save annotation file for validation
        # 검증용 주석 파일 저장
        with open("./final_annotation_val.txt", 'w', encoding='utf-8') as f:
            for line in cleaned_new_annos:
                f.write(line)
        print("finished")
    else:
        # Do not add extra helper data
        # STEP 1: modify config file
        # 추가 데이터를 사용하지 않는 경우
        # STEP 1: 설정 파일 수정
        with open("./configs/finetune_speaker.json", 'r', encoding='utf-8') as f:
            hps = json.load(f)

        # assign ids to new speakers
        # 새 스피커에 고유 ID 할당
        speaker2id = {}
        for i, speaker in enumerate(speakers):
            speaker2id[speaker] = i

        # modify n_speakers
        # 설정 파일의 스피커 수 수정
        hps['data']["n_speakers"] = len(speakers)

        # overwrite speaker names
        # 스피커 이름을 ID로 대체
        hps['speakers'] = speaker2id

        # 학습 관련 설정 수정 (로그 출력 주기, 평가 주기, 배치 크기 등)
        hps['train']['log_interval'] = 10
        hps['train']['eval_interval'] = 100
        hps['train']['batch_size'] = 16
        hps['data']['training_files'] = "final_annotation_train.txt"
        hps['data']['validation_files'] = "final_annotation_val.txt"
        
        # save modified config
        # 수정된 설정 파일 저장
        with open("./configs/modified_finetune_speaker.json", 'w', encoding='utf-8') as f:
            json.dump(hps, f, indent=2)

        # STEP 2: clean annotations, replace speaker names with assigned speaker IDs
        # STEP 2: 주석 데이터 정리 및 스피커 이름을 ID로 변환
        import text # 텍스트 클리닝을 위한 모듈

        cleaned_new_annos = []
        for i, line in enumerate(new_annos):
            path, speaker, txt = line.split("|")
            if len(txt) > 150: # 텍스트가 너무 긴 경우 건너뜀
                continue
            cleaned_text = text._clean_text(txt, hps['data']['text_cleaners']).replace("[ZH]", "") # 텍스트 클리닝 수행
            cleaned_text += "\n" if not cleaned_text.endswith("\n") else ""
            cleaned_new_annos.append(path + "|" + str(speaker2id[speaker]) + "|" + cleaned_text)

        final_annos = cleaned_new_annos

        # save annotation file
        # 최종 주석 파일 저장
        with open("./final_annotation_train.txt", 'w', encoding='utf-8') as f:
            for line in final_annos:
                f.write(line)
                
        # save annotation file for validation
        # 검증용 주석 파일 저장
        with open("./final_annotation_val.txt", 'w', encoding='utf-8') as f:
            for line in cleaned_new_annos:
                f.write(line)
        print("finished")
