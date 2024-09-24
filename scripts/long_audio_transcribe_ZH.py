from moviepy.editor import AudioFileClip
import whisper
import os
import json
import torchaudio
import librosa
import torch
import argparse

# 노이즈 제거된 오디오 파일이 저장된 디렉토리 경로
parent_dir = "./denoised_audio/"

# parent_dir 내의 파일 목록을 가져옴
filelist = list(os.walk(parent_dir))[0][2]


if __name__ == "__main__":
    # 명령줄 인자를 처리하기 위한 argparse 설정
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", default="CJE") # 사용할 언어 설정 (기본값: CJE)
    parser.add_argument("--whisper_size", default="medium") # Whisper 모델 크기 설정 (기본값: medium)
    args = parser.parse_args()

    # 선택한 언어에 따라 언어 토큰 설정
    if args.languages == "CJE":  # 중국어, 일본어, 영어 지원
        lang2token = {
            'zh': "[ZH]",
            'ja': "[JA]",
            "en": "[EN]",
        }
    elif args.languages == "CJ":  # 중국어, 일본어 지원
        lang2token = {
            'zh': "[ZH]",
            'ja': "[JA]",
        }
    elif args.languages == "C": # 중국어만 지원
        lang2token = {
            'zh': "[ZH]",
        }

    # GPU가 사용 가능한지 확인. GPU가 없으면 프로그램이 종료됨.
    assert(torch.cuda.is_available()), "Please enable GPU in order to run Whisper!"

    # 설정 파일에서 샘플링 레이트를 가져옴
    with open("./configs/finetune_speaker.json", 'r', encoding='utf-8') as f:
        hps = json.load(f)

    # 샘플링 레이트 설정
    target_sr = hps['data']['sampling_rate']

    # Whisper 모델 로드
    model = whisper.load_model(args.whisper_size)

    # 화자 정보 저장할 리스트
    speaker_annos = []

    # filelist 내의 각 파일에 대해 처리
    for file in filelist:
        print(f"transcribing {parent_dir + file}...\n") # 파일 처리 시작 메시지 출력

        # Whisper 모델을 사용할 때의 설정 (빔 사이즈, 결과 선택 옵션)
        options = dict(beam_size=5, best_of=5)
        transcribe_options = dict(task="transcribe", **options)

        # Whisper 모델을 사용해 오디오 파일을 텍스트로 변환 (단어 타임스탬프 포함)
        result = model.transcribe(parent_dir + file, word_timestamps=True, **transcribe_options)
        segments = result["segments"]

        # 결과에서 언어를 가져옴
        lang = result['language'] 

        # 지원하지 않는 언어의 경우 무시
        if result['language'] not in list(lang2token.keys()):
            print(f"{lang} not supported, ignoring...\n")
            continue
        
        # segment audio based on segment results
        # 파일명에서 캐릭터 이름과 코드 추출
        character_name = file.rstrip(".wav").split("_")[0]
        code = file.rstrip(".wav").split("_")[1]

        # 캐릭터 이름에 해당하는 디렉토리가 없으면 생성
        if not os.path.exists("./segmented_character_voice/" + character_name):
            os.mkdir("./segmented_character_voice/" + character_name)
        # 오디오 파일 로드 (노이즈 제거된 오디오 파일)
        wav, sr = torchaudio.load(parent_dir + file, frame_offset=0, num_frames=-1, normalize=True,
                                  channels_first=True)

        # 각 세그먼트(문장 단위) 처리
        for i, seg in enumerate(result['segments']):
            start_time = seg['start'] # 세그먼트 시작 시간
            end_time = seg['end'] # 세그먼트 끝 시간
            text = seg['text'] # Whisper 모델이 추출한 텍스트

            # 텍스트에 언어 토큰 추가
            text = lang2token[lang] + text.replace("\n", "") + lang2token[lang]
            text = text + "\n"

            # 오디오 파일에서 세그먼트의 오디오 부분을 추출
            wav_seg = wav[:, int(start_time*sr):int(end_time*sr)]

            # 세그먼트 파일명 생성
            wav_seg_name = f"{character_name}_{code}_{i}.wav"

            # 파일 저장 경로 설정
            savepth = "./segmented_character_voice/" + character_name + "/" + wav_seg_name

            # 화자 정보(파일 경로, 캐릭터 이름, 텍스트)를 speaker_annos 리스트에 추가
            speaker_annos.append(savepth + "|" + character_name + "|" + text)
            print(f"Transcribed segment: {speaker_annos[-1]}")
            # trimmed_wav_seg = librosa.effects.trim(wav_seg.squeeze().numpy())
            # trimmed_wav_seg = torch.tensor(trimmed_wav_seg[0]).unsqueeze(0)

            # 추출한 오디오 세그먼트를 지정된 경로에 저장
            torchaudio.save(savepth, wav_seg, target_sr, channels_first=True)

    # 처리한 세그먼트가 없을 경우 경고 메시지 출력
    if len(speaker_annos) == 0:
        print("Warning: no long audios & videos found, this IS expected if you have only uploaded short audios")
        print("this IS NOT expected if you have uploaded any long audios, videos or video links. Please check your file structure or make sure your audio/video language is supported.")
        
    # 화자 정보(speaker_annos)를 텍스트 파일에 저장
    with open("./long_character_anno.txt", 'w', encoding='utf-8') as f:
        for line in speaker_annos:
            f.write(line)

# 아래는 처리 결과
# ./segmented_character_voice/masiro/masiro_010_0.wav|masiro|[KO] 안녕하세요 여러분들 깜짝 놀랐죠?[KO]
# ./segmented_character_voice/masiro/masiro_010_1.wav|masiro|[KO] 시로입니다![KO]
# ./segmented_character_voice/masiro/masiro_010_2.wav|masiro|[KO] 되게 오랜만이죠?[KO]