import whisper
import os
import json
import torchaudio
import argparse
import torch

# 언어별 토큰 정의
lang2token = {
    'zh': "[ZH]",  # 중국어
    'ja': "[JA]",  # 일본어
    'en': "[EN]",  # 영어
    'ko': "[KO]",  # 한국어
}

# 오디오 파일을 변환하고 텍스트로 변환하는 함수
def transcribe_one(audio_path):
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio).to(model.device)
#     print(f"Mel spectrogram shape: {mel.shape}")

    _, probs = model.detect_language(mel)
    lang = "ko"
#     print(f"Detected language: {lang}")

    # 디코딩 옵션에서 언어를 "ko"로 설정
    options = whisper.DecodingOptions(beam_size=5, language=lang)
    result = whisper.decode(model, mel, options)

    print(result.text)
    return lang, result.text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", default="CJKE")  # 처리할 언어 설정
    parser.add_argument("--whisper_size", default="medium")  # Whisper 모델 크기 설정
    args = parser.parse_args()

    # 선택된 언어에 따라 lang2token 초기화
    if args.languages == "CJKE":
        lang2token = {'zh': "[ZH]", 'ja': "[JA]", 'en': "[EN]", 'ko': "[KO]"}
    elif args.languages == "CJE":
        lang2token = {'zh': "[ZH]", 'ja': "[JA]", 'en': "[EN]"}
    elif args.languages == "CJ":
        lang2token = {'zh': "[ZH]", 'ja': "[JA]"}
    elif args.languages == "C":
        lang2token = {'zh': "[ZH]"}

    # GPU 사용 여부 확인
    assert torch.cuda.is_available(), "Please enable GPU in order to run Whisper!"

    # Whisper 모델 로드
    model = whisper.load_model(args.whisper_size)

    # 오디오 파일이 있는 디렉토리 설정
    parent_dir = "./custom_character_voice/"
    speaker_names = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
    print('Speaker names:', speaker_names)

    speaker_annos = []  # 텍스트 데이터를 저장할 리스트
    total_files = sum([len(files) for _, _, files in os.walk(parent_dir)])  # 전체 파일 수 계산
    print('Total files:', total_files)

    # 샘플링 레이트 가져오기
    with open("./configs/finetune_speaker.json", 'r', encoding='utf-8') as f:
        hps = json.load(f)
    target_sr = hps['data']['sampling_rate']

    processed_files = 0

    # 각 스피커 디렉토리 내 오디오 파일 처리
    for speaker in speaker_names:
        speaker_path = os.path.join(parent_dir, speaker)
        for i, wavfile in enumerate(os.listdir(speaker_path)):
            if not wavfile.endswith(".wav") or wavfile.startswith("processed_"):
                continue
            try:
                # 오디오 파일 로드
                wav_path = os.path.join(speaker_path, wavfile)
                wav, sr = torchaudio.load(wav_path)
                
                # 멀티채널 데이터를 모노로 변환
                wav = wav.mean(dim=0).unsqueeze(0)

                # 샘플링 레이트와 채널 처리
                if sr != target_sr:
                    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
                    wav = resampler(wav)

                # Resample 후 데이터 크기 확인
#                 print(f"Shape after resampling: {wav.shape}")

                # 길이 초과 파일 무시
                if wav.shape[1] / target_sr > 20:  # Resampling 후 target_sr 사용
                    print(f"Skipping {wavfile} (too long)")
                    continue

                # 처리된 파일 저장
                save_path = os.path.join(speaker_path, f"processed_{i}.wav")
                torchaudio.save(save_path, wav, target_sr)

                # 텍스트 변환
                lang, text = transcribe_one(save_path)

                if lang not in lang2token:
                    print(f"Unsupported language: {lang}, skipping {wavfile}")
                    continue

                # 결과 저장
                text = f"{lang2token[lang]}{text}{lang2token[lang]}\n"
                speaker_annos.append(f"{save_path}|{speaker}|{text}")

                processed_files += 1
                print(f"Processed: {processed_files}/{total_files}")

            except Exception as e:
                print(f"Error processing file {wavfile}: {e}")


    # 결과 저장
    if not speaker_annos:
        print("No valid files found. Check your input data.")

    with open("short_character_anno.txt", 'w', encoding='utf-8') as f:
        f.writelines(speaker_annos)
