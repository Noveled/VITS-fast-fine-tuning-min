import whisper
import os
import json
import torchaudio
import argparse
import torch

# 언어별 토큰을 정의하는 사전 (중국어, 일본어, 영어, 한국어)
lang2token = {
            'zh': "[ZH]",  # 중국어
            'ja': "[JA]",  # 일본어
            'en': "[EN]",  # 영어
            'ko': "[KO]",  # 한국어
        }

# 오디오 파일을 변환하고 텍스트로 변환하는 함수
def transcribe_one(audio_path):
    # 오디오 파일을 로드하고 길이를 30초로 맞춤
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)

    # 로그-멜 스펙트로그램을 만들고, 모델이 위치한 장치로 이동
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # 언어 감지
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")  # 감지된 언어 출력
    lang = max(probs, key=probs.get)  # 확률이 가장 높은 언어를 선택

    # 오디오 디코딩 옵션 설정
    options = whisper.DecodingOptions(beam_size=5)
    result = whisper.decode(model, mel, options)  # 오디오를 텍스트로 변환

    # 변환된 텍스트 출력
    print(result.text)
    return lang, result.text  # 감지된 언어와 변환된 텍스트를 반환

# 메인 실행 부분
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", default="CJKE")  # 처리할 언어 설정
    parser.add_argument("--whisper_size", default="medium")  # Whisper 모델 크기 설정
    args = parser.parse_args()

    # 선택된 언어에 따라 lang2token 사전을 설정
    if args.languages == "CJKE":
        lang2token = {
            'zh': "[ZH]",
            'ja': "[JA]",
            "en": "[EN]",
	        "ko": "[KO]",
        }
    elif args.languages == "CJE":
        lang2token = {
            'zh': "[ZH]",
            'ja': "[JA]",
            "en": "[EN]",
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
    assert (torch.cuda.is_available()), "Please enable GPU in order to run Whisper!"
    
    # Whisper 모델 로드
    model = whisper.load_model(args.whisper_size)

    # 오디오 파일이 있는 디렉토리 설정
    parent_dir = "./custom_character_voice/"
    speaker_names = list(os.walk(parent_dir))[0][1]  # 디렉토리 내 스피커 이름 리스트
    speaker_annos = []  # 스피커와 텍스트 데이터를 저장할 리스트
    total_files = sum([len(files) for r, d, files in os.walk(parent_dir)])  # 전체 파일 수 계산

    # 타겟 샘플링 레이트 가져오기 (설정 파일에서 불러옴)
    with open("./configs/finetune_speaker.json", 'r', encoding='utf-8') as f:
        hps = json.load(f)
    target_sr = hps['data']['sampling_rate']  # 타겟 샘플링 레이트 설정

    processed_files = 0  # 처리된 파일 수 초기화

    # 각 스피커 디렉토리 내 오디오 파일을 순차적으로 처리
    for speaker in speaker_names:
        for i, wavfile in enumerate(list(os.walk(parent_dir + speaker))[0][2]):
            # 이미 처리된 파일은 건너뜀
            if wavfile.startswith("processed_"):
                continue
            try:
                # 오디오 파일 로드
                wav, sr = torchaudio.load(parent_dir + speaker + "/" + wavfile, frame_offset=0, num_frames=-1, normalize=True,
                                          channels_first=True)
                wav = wav.mean(dim=0).unsqueeze(0)  # 모노로 변환
                
                # 타겟 샘플링 레이트와 다르면 리샘플링
                if sr != target_sr:
                    wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(wav)

                # 파일이 20초 이상일 경우 처리하지 않음
                if wav.shape[1] / sr > 20:
                    print(f"{wavfile} too long, ignoring\n")
                
                # 처리된 파일로 저장
                save_path = parent_dir + speaker + "/" + f"processed_{i}.wav"
                torchaudio.save(save_path, wav, target_sr, channels_first=True)

                # 텍스트로 변환
                lang, text = transcribe_one(save_path)

                # 지원되지 않는 언어는 무시
                if lang not in list(lang2token.keys()):
                    print(f"{lang} not supported, ignoring\n")
                    continue

                # 텍스트에 언어 마크 추가 및 저장
                text = lang2token[lang] + text + lang2token[lang] + "\n"
                speaker_annos.append(save_path + "|" + speaker + "|" + text)

                processed_files += 1  # 처리된 파일 수 증가
                print(f"Processed: {processed_files}/{total_files}")  # 처리 상태 출력
            except:
                continue  # 오류 발생 시 해당 파일은 건너뜀

    # 짧은 오디오가 없을 경우 경고 메시지 출력
    if len(speaker_annos) == 0:
        print("Warning: no short audios found, this IS expected if you have only uploaded long audios, videos or video links.")
        print("this IS NOT expected if you have uploaded a zip file of short audios. Please check your file structure or make sure your audio language is supported.")
    
    # 어노테이션 파일에 결과 저장
    with open("short_character_anno.txt", 'w', encoding='utf-8') as f:
        for line in speaker_annos:
            f.write(line)
