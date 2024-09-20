import torch
import argparse
import json

if __name__ == "__main__":
    # 명령줄 인자 처리기 생성
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="./OUTPUT_MODEL/G_latest.pth") # 모델 파일 경로 설정
    parser.add_argument("--config_dir", type=str, default="./configs/modified_finetune_speaker.json") # 설정 파일 경로 설정
    args = parser.parse_args()

    # 학습된 모델의 state_dict를 로드
    model_sd = torch.load(args.model_dir, map_location='cpu') # 모델 파라미터를 CPU에 로드
    # 설정 파일을 로드하여 하이퍼파라미터 및 스피커 정보 가져오기
    with open(args.config_dir, 'r', encoding='utf-8') as f:
        hps = json.load(f)

    # 설정 파일에서 유효한 스피커 목록을 가져옴
    valid_speakers = list(hps['speakers'].keys())

    # 스피커의 수가 기존 설정 파일에 있는 스피커 수보다 적은 경우
    if hps['data']['n_speakers'] > len(valid_speakers):
        # 새로운 스피커 임베딩을 저장할 텐서를 생성 (각 스피커에 대해 256차원 임베딩)
        new_emb_g = torch.zeros([len(valid_speakers), 256])

        # 기존 모델에서 사용되던 스피커 임베딩 가져오기
        old_emb_g = model_sd['model']['emb_g.weight']

        # 유효한 스피커에 대해서만 임베딩을 다시 할당
        for i, speaker in enumerate(valid_speakers):
            # 기존 모델의 스피커 임베딩을 새로 생성된 임베딩에 복사
            new_emb_g[i, :] = old_emb_g[hps['speakers'][speaker], :]
            # 스피커 ID를 재할당 (인덱스 값으로 덮어쓰기)
            hps['speakers'][speaker] = i

        # 설정 파일에서 스피커 수를 유효한 스피커의 수로 수정
        hps['data']['n_speakers'] = len(valid_speakers)
        # 모델 파라미터에서 새로운 스피커 임베딩을 저장
        model_sd['model']['emb_g.weight'] = new_emb_g

        # 수정된 설정 파일을 저장
        with open("./finetune_speaker.json", 'w', encoding='utf-8') as f:
            json.dump(hps, f, indent=2)
        # 수정된 모델 파라미터를 저장
        torch.save(model_sd, "./G_latest.pth")
    else: # 유효한 스피커 수가 기존 설정된 스피커 수보다 적거나 같을 경우
        # 설정 파일을 그대로 저장
        with open("./finetune_speaker.json", 'w', encoding='utf-8') as f:
            json.dump(hps, f, indent=2)
        # 모델 파라미터를 그대로 저장
        torch.save(model_sd, "./G_latest.pth")

    # save another config file copy in MoeGoe format
    # MoeGoe 포맷에 맞춘 설정 파일을 생성하여 스피커 이름을 그대로 유지
    hps['speakers'] = valid_speakers # MoeGoe는 스피커 ID 대신 이름을 사용
    with open("./moegoe_config.json", 'w', encoding='utf-8') as f:
        json.dump(hps, f, indent=2) # MoeGoe 포맷의 설정 파일 저장



