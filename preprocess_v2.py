import os
import argparse
import json
import sys

# 파이썬의 재귀 깊이 제한을 늘려서 RecursionError 방지
sys.setrecursionlimit(500000)

if __name__ == "__main__":
    # 명령행 인자 파싱을 위한 argparse 객체 생성
    parser = argparse.ArgumentParser()
    
    # 추가 데이터 사용 여부를 선택하는 인자
    parser.add_argument("--add_auxiliary_data", type=bool, help="Whether to add extra data as fine-tuning helper")
    
    # 처리할 언어 옵션 (기본값: "CJKE")
    parser.add_argument("--languages", default="CJKE")
    
    # 명령행 인자를 파싱
    args = parser.parse_args()

    # 언어에 따른 언어 코드 설정
    if args.languages == "CJKE":
        langs = ["[ZH]", "[JA]", "[EN]", "[KO]"]  # 중국어, 일본어, 영어, 한국어
    elif args.languages == "CJE":
        langs = ["[ZH]", "[JA]", "[EN]"]  # 중국어, 일본어, 영어
    elif args.languages == "CJ":
        langs = ["[ZH]", "[JA]"]  # 중국어, 일본어
    elif args.languages == "C":
        langs = ["[ZH]"]  # 중국어만

    new_annos = []
    
    # 소스 1: 짧은 오디오의 텍스트 변환 어노테이션 불러오기
    if os.path.exists("short_character_anno.txt"):
        with open("short_character_anno.txt", 'r', encoding='utf-8') as f:
            short_character_anno = f.readlines()
            new_annos += short_character_anno
    
    # 소스 2: 긴 오디오 세그먼트의 텍스트 변환 어노테이션 불러오기
    if os.path.exists("./long_character_anno.txt"):
        with open("./long_character_anno.txt", 'r', encoding='utf-8') as f:
            long_character_anno = f.readlines()
            new_annos += long_character_anno

    # 모든 스피커 이름을 추출
    speakers = []
    for line in new_annos:
        path, speaker, text = line.split("|")
        if speaker not in speakers:
            speakers.append(speaker)

    # 스피커가 하나도 없을 경우 예외 처리
    assert (len(speakers) != 0), "No audio file found. Please check your uploaded file structure."
    
    # 추가적인 학습 보조 데이터 사용 여부에 따라 처리
    if args.add_auxiliary_data:
        # 소스 3 (선택 사항): 추가 학습을 위한 샘플 오디오 데이터 불러오기
        with open("./sampled_audio4ft.txt", 'r', encoding='utf-8') as f:
            old_annos = f.readlines()
        
        # 선택된 언어에 해당하는 어노테이션만 필터링
        filtered_old_annos = []
        for line in old_annos:
            for lang in langs:
                if lang in line:
                    filtered_old_annos.append(line)
        old_annos = filtered_old_annos
        
        # 스피커 이름 업데이트
        for line in old_annos:
            path, speaker, text = line.split("|")
            if speaker not in speakers:
                speakers.append(speaker)
        
        num_old_voices = len(old_annos)  # 이전 오디오 데이터의 수
        num_new_voices = len(new_annos)  # 새로 추가된 오디오 데이터의 수
        
        # 1단계: 새로운 데이터와 이전 데이터의 비율을 맞춤
        cc_duplicate = num_old_voices // num_new_voices
        if cc_duplicate == 0:
            cc_duplicate = 1

        # 2단계: 설정 파일 수정
        with open("./configs/finetune_speaker.json", 'r', encoding='utf-8') as f:
            hps = json.load(f)

        # 스피커에 새로운 ID 할당
        speaker2id = {}
        for i, speaker in enumerate(speakers):
            speaker2id[speaker] = i

        # 스피커 수를 수정
        hps['data']["n_speakers"] = len(speakers)

        # 스피커 이름을 새로 할당된 ID로 덮어쓰기
        hps['speakers'] = speaker2id
        
        # 학습 설정 업데이트
        hps['train']['log_interval'] = 10
        hps['train']['eval_interval'] = 100
        hps['train']['batch_size'] = 16
        hps['data']['training_files'] = "final_annotation_train.txt"
        hps['data']['validation_files'] = "final_annotation_val.txt"
        
        # 수정된 설정 파일 저장
        with open("./configs/modified_finetune_speaker.json", 'w', encoding='utf-8') as f:
            json.dump(hps, f, indent=2)

        # 3단계: 어노테이션을 정리하고 스피커 이름을 ID로 대체
        import text
        cleaned_new_annos = []
        for i, line in enumerate(new_annos):
            path, speaker, txt = line.split("|")
            if len(txt) > 150:  # 텍스트 길이가 150자를 넘으면 생략
                continue
            cleaned_text = text._clean_text(txt, hps['data']['text_cleaners'])
            cleaned_text += "\n" if not cleaned_text.endswith("\n") else ""
            cleaned_new_annos.append(path + "|" + str(speaker2id[speaker]) + "|" + cleaned_text)
        
        cleaned_old_annos = []
        for i, line in enumerate(old_annos):
            path, speaker, txt = line.split("|")
            if len(txt) > 150:
                continue
            cleaned_text = text._clean_text(txt, hps['data']['text_cleaners'])
            cleaned_text += "\n" if not cleaned_text.endswith("\n") else ""
            cleaned_old_annos.append(path + "|" + str(speaker2id[speaker]) + "|" + cleaned_text)
        
        # 새로운 어노테이션과 이전 어노테이션 병합
        final_annos = cleaned_old_annos + cc_duplicate * cleaned_new_annos
        
        # 학습용 어노테이션 파일 저장
        with open("./final_annotation_train.txt", 'w', encoding='utf-8') as f:
            for line in final_annos:
                f.write(line)
        
        # 검증용 어노테이션 파일 저장
        with open("./final_annotation_val.txt", 'w', encoding='utf-8') as f:
            for line in cleaned_new_annos:
                f.write(line)
        
        print("finished")
    
    else:
        # 추가 데이터를 사용하지 않는 경우

        # 1단계: 설정 파일 수정
        with open("./configs/finetune_speaker.json", 'r', encoding='utf-8') as f:
            hps = json.load(f)

        # 스피커에 새로운 ID 할당
        speaker2id = {}
        for i, speaker in enumerate(speakers):
            speaker2id[speaker] = i

        # 스피커 수 수정
        hps['data']["n_speakers"] = len(speakers)
        hps['speakers'] = speaker2id

        # 학습 설정 업데이트
        hps['train']['log_interval'] = 10
        hps['train']['eval_interval'] = 100
        hps['train']['batch_size'] = 16
        hps['data']['training_files'] = "final_annotation_train.txt"
        hps['data']['validation_files'] = "final_annotation_val.txt"

        # 수정된 설정 파일 저장
        with open("./configs/modified_finetune_speaker.json", 'w', encoding='utf-8') as f:
            json.dump(hps, f, indent=2)

        # 2단계: 어노테이션 정리 및 스피커 이름을 ID로 대체
        import text

        cleaned_new_annos = []
        for i, line in enumerate(new_annos):
            path, speaker, txt = line.split("|")
            if len(txt) > 150:
                continue
            cleaned_text = text._clean_text(txt, hps['data']['text_cleaners']).replace("[ZH]", "")
            cleaned_text += "\n" if not cleaned_text.endswith("\n") else ""
            cleaned_new_annos.append(path + "|" + str(speaker2id[speaker]) + "|" + cleaned_text)

        final_annos = cleaned_new_annos

        # 학습용 어노테이션 파일 저장
        with open("./final_annotation_train.txt", 'w', encoding='utf-8') as f:
            for line in final_annos:
                f.write(line)

        # 검증용 어노테이션 파일 저장
        with open("./final_annotation_val.txt", 'w', encoding='utf-8') as f:
            for line in cleaned_new_annos:
                f.write(line)

        print("finished")
