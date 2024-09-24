from moviepy.editor import AudioFileClip
import whisper
import os
import json
import torchaudio
import librosa
import torch
import argparse

# 오디오 파일이 저장된 디렉토리 설정
parent_dir = "./denoised_audio/"
filelist = list(os.walk(parent_dir))[0][2]  # 디렉토리 내 파일 목록을 가져옴

if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # 명령행 인자 처리를 위한 파서
    parser.add_argument("--languages", default="CJKE")  # 처리할 언어 설정 (기본값: "CJKE")
    parser.add_argument("--whisper_size", default="medium")  # Whisper 모델 크기 설정 (기본값: "medium")
    args = parser.parse_args()

    # 처리할 언어에 따라 언어 토큰 설정
    if args.languages == "CJKE":
        lang2token = {
            'zh': "[ZH]",  # 중국어
            'ja': "[JA]",  # 일본어
            'en': "[EN]",  # 영어
            'ko': "[KO]",  # 한국어
        }
    elif args.languages == "CJE":
        lang2token = {
            'zh': "[ZH]",
            'ja': "[JA]",
            'en': "[EN]",
        }
    elif args.languages == "CJ":
        lang2token = {
            'zh': "[ZH]",
            'ja': "[JA]",
        }
    elif args.languages == "C":
        lang2token = {
            'zh': "[ZH]",
        }
    
    # GPU가 사용 가능한지 확인
    assert(torch.cuda.is_available()), "Please enable GPU in order to run Whisper!"

    # Whisper 모델을 로드하고 타겟 샘플링 레이트를 설정
    with open("./configs/finetune_speaker.json", 'r', encoding='utf-8') as f:
        hps = json.load(f)
    target_sr = hps['data']['sampling_rate']  # 샘플링 레이트 설정
    model = whisper.load_model(args.whisper_size)  # Whisper 모델 로드

    speaker_annos = []  # 스피커 및 텍스트 어노테이션을 저장할 리스트

    # 파일 리스트를 순회하며 처리
    for file in filelist:
        print(f"transcribing {parent_dir + file}...\n")  # 처리 중인 파일 출력

        # Whisper 모델을 사용한 음성 변환 옵션 설정
        options = dict(beam_size=5, best_of=5)
        transcribe_options = dict(task="transcribe", **options)

        # 오디오 파일을 텍스트로 변환
        result = model.transcribe(parent_dir + file, word_timestamps=True, **transcribe_options)
        segments = result["segments"]  # 텍스트 세그먼트 정보
        lang = result['language']  # 감지된 언어

        # 감지된 언어가 지원되지 않는 경우 건너뜀
        if result['language'] not in list(lang2token.keys()):
            print(f"{lang} not supported, ignoring...\n")
            continue

        # 파일명에서 캐릭터 이름과 코드 추출
        character_name = file.rstrip(".wav").split("_")[0]
        code = file.rstrip(".wav").split("_")[1]

        # 캐릭터의 오디오 세그먼트를 저장할 디렉토리가 없으면 생성
        if not os.path.exists("./segmented_character_voice/" + character_name):
            os.mkdir("./segmented_character_voice/" + character_name)

        # 오디오 파일 로드
        wav, sr = torchaudio.load(parent_dir + file, frame_offset=0, num_frames=-1, normalize=True, channels_first=True)

        # 텍스트 세그먼트 별로 오디오 파일을 분할
        for i, seg in enumerate(result['segments']):
            start_time = seg['start']  # 세그먼트 시작 시간
            end_time = seg['end']  # 세그먼트 종료 시간
            text = seg['text']  # 세그먼트의 텍스트

            # 텍스트 앞뒤에 언어 토큰 추가
            text = lang2token[lang] + text.replace("\n", "") + lang2token[lang] + "\n"
            
            # 오디오 세그먼트를 잘라냄
            wav_seg = wav[:, int(start_time * sr):int(end_time * sr)]
            wav_seg_name = f"{character_name}_{code}_{i}.wav"  # 저장할 세그먼트 파일명
            savepth = "./segmented_character_voice/" + character_name + "/" + wav_seg_name  # 저장 경로
            
            # 어노테이션 추가
            speaker_annos.append(savepth + "|" + character_name + "|" + text)
            print(f"Transcribed segment: {speaker_annos[-1]}")  # 처리된 세그먼트 출력

            # 세그먼트 오디오를 저장
            torchaudio.save(savepth, wav_seg, target_sr, channels_first=True)

    # 처리된 파일이 없는 경우 경고 메시지 출력
    if len(speaker_annos) == 0:
        print("Warning: no long audios & videos found, this IS expected if you have only uploaded short audios")
        print("this IS NOT expected if you have uploaded any long audios, videos or video links. Please check your file structure or make sure your audio/video language is supported.")
    
    # 어노테이션을 파일에 저장
    with open("./long_character_anno.txt", 'w', encoding='utf-8') as f:
        for line in speaker_annos:
            f.write(line)
