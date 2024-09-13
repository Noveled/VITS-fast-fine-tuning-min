import os
import json
import torchaudio

# 원본 오디오 파일이 저장된 디렉토리 경로
raw_audio_dir = "./raw_audio/"

# 노이즈 제거된 오디오 파일을 저장할 디렉토리 경로
denoise_audio_dir = "./denoised_audio/"

# raw_audio_dir 디렉토리 내의 파일 목록을 가져옴
filelist = list(os.walk(raw_audio_dir))[0][2]

# 2023/4/21: Get the target sampling rate
# 2023/4/21: 타겟 샘플링 레이트를 얻기 위해 설정 파일을 읽음
with open("./configs/finetune_speaker.json", 'r', encoding='utf-8') as f:
    hps = json.load(f)

# 설정 파일에서 샘플링 레이트 값을 추출
target_sr = hps['data']['sampling_rate']

# filelist의 각 파일에 대해 처리
for file in filelist:
    # 파일 확장자가 ".wav"인 경우에만 실행
    if file.endswith(".wav"):
        # Demucs를 사용해 오디오에서 보컬 트랙과 다른 트랙을 분리
        os.system(f"demucs --two-stems=vocals {raw_audio_dir}{file}")

# 분리된 파일들을 처리
for file in filelist:
    # ".wav" 확장자를 제거하여 파일 이름만 남김
    file = file.replace(".wav", "")

    # 분리된 보컬 트랙을 로드 (separated/htdemucs 폴더에서 해당 파일을 찾음)
    wav, sr = torchaudio.load(f"./separated/htdemucs/{file}/vocals.wav", frame_offset=0, num_frames=-1, normalize=True,
                              channels_first=True)
    # merge two channels into one
    # 두 개의 채널(스테레오)을 하나의 채널로 병합 (단일 채널로 변환)
    wav = wav.mean(dim=0).unsqueeze(0)

    # 원본 샘플링 레이트(sr)가 타겟 샘플링 레이트와 다르면 리샘플링 수행
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(wav)
    # 노이즈 제거된 오디오 파일을 denoise_audio_dir에 저장, 이름은 원래 파일명으로 저장
    torchaudio.save(denoise_audio_dir + file + ".wav", wav, target_sr, channels_first=True)