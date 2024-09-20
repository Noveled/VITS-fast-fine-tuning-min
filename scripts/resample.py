import os
import json
import argparse
import torchaudio


def main():
    # 설정 파일(finute_speaker.json)을 열어 샘플링 레이트 정보를 가져옵니다.
    with open("./configs/finetune_speaker.json", 'r', encoding='utf-8') as f:
        hps = json.load(f)

    # 설정 파일에서 'sampling_rate' 값을 읽어와서 target_sr 변수에 저장합니다.
    target_sr = hps['data']['sampling_rate']

    # "./sampled_audio4ft" 디렉토리 안의 파일 목록을 가져옵니다.
    # 아마도 샘플 캐릭터 보이스 폴더임
    filelist = list(os.walk("./sampled_audio4ft"))[0][2]

    # target_sr(타겟 샘플링 레이트)가 22050Hz와 다를 경우에만 오디오 파일의 샘플링 레이트를 변경합니다.
    if target_sr != 22050:
        # 파일 리스트 내의 각 오디오 파일을 순회하며 처리
        for wavfile in filelist:
            # 오디오 파일을 로드합니다. (frame_offset: 시작 프레임, num_frames: 프레임 수, normalize: 값 정규화)
            wav, sr = torchaudio.load("./sampled_audio4ft" + "/" + wavfile, frame_offset=0, num_frames=-1,
                                      normalize=True, channels_first=True)
            # 현재 파일의 샘플링 레이트(sr)가 target_sr과 다를 경우 Resample을 사용하여 샘플링 레이트를 변경합니다.
            wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(wav)

            # 샘플링 레이트가 변경된 오디오 파일을 원래 경로에 다시 저장합니다
            torchaudio.save("./sampled_audio4ft" + "/" + wavfile, wav, target_sr, channels_first=True)

# 메인 함수 실행
if __name__ == "__main__":
    main()