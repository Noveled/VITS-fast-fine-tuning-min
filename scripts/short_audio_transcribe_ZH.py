import whisper
import os
import json
import torchaudio
import argparse
import torch

# 언어별 토큰을 정의하는 딕셔너리
lang2token = {
            'zh': "[ZH]",
            'ja': "[JA]",
            "en": "[EN]",
        }

# 단일 오디오 파일을 처리하는 함수
def transcribe_one(audio_path):
    # load audio and pad/trim it to fit 30 seconds
    # 오디오 파일을 로드하고 30초에 맞춰 패딩하거나 자릅니다.
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    # 로그-멜 스펙트로그램을 생성하고 모델이 위치한 장치로 이동시킵니다 (CPU 또는 GPU).
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    # 음성에서 사용된 언어를 감지합니다.
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}") # 감지된 언어 출력
    lang = max(probs, key=probs.get)

    # decode the audio
    # 오디오를 디코딩합니다
    options = whisper.DecodingOptions(beam_size=5) # 빔 크기를 5로 설정
    result = whisper.decode(model, mel, options)

    # print the recognized text
    # 인식된 텍스트를 출력합니다.
    print(result.text)

    # 언어와 텍스트를 반환합니다.
    return lang, result.text

# 메인 실행 코드
if __name__ == "__main__":
    # 명령줄 인자를 처리합니다.
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", default="CJE") # 사용할 언어 그룹 지정 (기본값 CJE: 중국어, 일본어, 영어)
    parser.add_argument("--whisper_size", default="medium") # Whisper 모델 크기 지정 (기본값 medium)
    args = parser.parse_args()

    # 언어 그룹에 따라 lang2token 딕셔너리를 수정합니다.
    if args.languages == "CJE":
        lang2token = {
            'zh': "[ZH]", # 중국어
            'ja': "[JA]", # 일본어
            "en": "[EN]", # 영어
        }
    elif args.languages == "CJ":
        lang2token = {
            'zh': "[ZH]", # 중국어
            'ja': "[JA]", # 일본어
        }
    elif args.languages == "C":
        lang2token = {
            'zh': "[ZH]", # 중국어
        }

    # GPU가 사용 가능한지 확인하고, 불가능하면 에러를 발생시킵니다.
    assert (torch.cuda.is_available()), "Please enable GPU in order to run Whisper!"

    # Whisper 모델을 로드합니다. 인자로 받은 모델 크기를 사용합니다.
    model = whisper.load_model(args.whisper_size)

    # 오디오 파일들이 있는 디렉터리를 설정합니다.
    parent_dir = "./custom_character_voice/"
    # 디렉터리 안에 있는 스피커 이름(하위 폴더 이름)을 가져옵니다.
    speaker_names = list(os.walk(parent_dir))[0][1]

    # 스피커의 주석 데이터를 저장할 리스트를 초기화합니다.
    speaker_annos = []

    # 총 오디오 파일 수를 계산합니다.
    total_files = sum([len(files) for r, d, files in os.walk(parent_dir)])

    # resample audios
    # 2023/4/21: Get the target sampling rate
    # 오디오 파일을 다시 샘플링하는 과정
    # 2023/4/21: 타겟 샘플링 레이트를 설정합니다.
    with open("./configs/finetune_speaker.json", 'r', encoding='utf-8') as f:
        hps = json.load(f)

    # 설정 파일에서 타겟 샘플링 레이트를 가져옵니다.
    target_sr = hps['data']['sampling_rate']

    # 처리된 파일 수를 추적하는 변수입니다.
    processed_files = 0

    # 각 스피커 폴더에 대해 처리합니다.
    for speaker in speaker_names:
        # 각 스피커 폴더 안의 파일을 순회합니다.
        for i, wavfile in enumerate(list(os.walk(parent_dir + speaker))[0][2]):
            # try to load file as audio
            # 이미 처리된 파일("processed_"로 시작하는 파일)은 건너뜁니다.
            if wavfile.startswith("processed_"):
                continue
            try:
                # 오디오 파일을 로드하고, 여러 채널이 있을 경우 평균을 계산하여 모노로 만듭니다.
                wav, sr = torchaudio.load(parent_dir + speaker + "/" + wavfile, frame_offset=0, num_frames=-1, normalize=True, channels_first=True)
                wav = wav.mean(dim=0).unsqueeze(0)

                # 샘플링 레이트가 타겟과 다르면 다시 샘플링합니다.
                if sr != target_sr:
                    wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(wav)
                
                # 오디오가 20초 이상이면 건너뜁니다.
                if wav.shape[1] / sr > 20:
                    print(f"{wavfile} too long, ignoring\n")
                save_path = parent_dir + speaker + "/" + f"processed_{i}.wav"

                # 처리된 오디오 파일을 저장할 경로를 설정합니다.
                torchaudio.save(save_path, wav, target_sr, channels_first=True)

                # transcribe text
                # 텍스트를 트랜스크립션합니다.
                lang, text = transcribe_one(save_path)

                # 감지된 언어가 지원되지 않는 경우 건너뜁니다.
                if lang not in list(lang2token.keys()):
                    print(f"{lang} not supported, ignoring\n")
                    continue

                # 텍스트에 언어 토큰을 추가하고, 스피커 주석에 추가합니다.
                text = lang2token[lang] + text + lang2token[lang] + "\n"
                speaker_annos.append(save_path + "|" + speaker + "|" + text)
                
                # 처리된 파일 수를 업데이트하고 상태를 출력합니다.
                processed_files += 1
                print(f"Processed: {processed_files}/{total_files}")
            except:
                # 예외가 발생하면 해당 파일을 건너뜁니다.
                continue

    # # clean annotation
    # import argparse
    # import text
    # from utils import load_filepaths_and_text
    # for i, line in enumerate(speaker_annos):
    #     path, sid, txt = line.split("|")
    #     cleaned_text = text._clean_text(txt, ["cjke_cleaners2"])
    #     cleaned_text += "\n" if not cleaned_text.endswith("\n") else ""
    #     speaker_annos[i] = path + "|" + sid + "|" + cleaned_text
    # write into annotation

    # 처리된 짧은 오디오 파일이 없는 경우 경고 메시지를 출력합니다.
    if len(speaker_annos) == 0:
        print("Warning: no short audios found, this IS expected if you have only uploaded long audios, videos or video links.")
        print("this IS NOT expected if you have uploaded a zip file of short audios. Please check your file structure or make sure your audio language is supported.")
    
    # 최종적으로 주석 데이터를 파일로 저장합니다.
    with open("short_character_anno.txt", 'w', encoding='utf-8') as f:
        for line in speaker_annos:
            f.write(line)


    # 이후, 필요한 경우 설정 파일을 수정하는 코드 부분이 주석 처리되어 있습니다.
    
    # import json
    # # generate new config
    # with open("./configs/finetune_speaker.json", 'r', encoding='utf-8') as f:
    #     hps = json.load(f)
    # # modify n_speakers
    # hps['data']["n_speakers"] = 1000 + len(speaker2id)
    # # add speaker names
    # for speaker in speaker_names:
    #     hps['speakers'][speaker] = speaker2id[speaker]
    # # save modified config
    # with open("./configs/modified_finetune_speaker.json", 'w', encoding='utf-8') as f:
    #     json.dump(hps, f, indent=2)
    # print("finished")
