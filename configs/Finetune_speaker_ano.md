# Finetune Speaker Configuration

이 문서는 `finetune_speaker.json` 파일에서 사용하는 설정 항목들을 설명합니다. 이 파일은 음성 합성 모델을 훈련할 때 필요한 설정 정보를 담고 있습니다.

## 1. 학습 설정 (`train`)

- **log_interval**: 학습 과정에서 로그를 출력할 단계 간격. (`200`)
- **eval_interval**: 검증을 수행할 단계 간격. (`1000`)
- **seed**: 재현성을 보장하기 위한 랜덤 시드. (`1234`)
- **epochs**: 학습할 총 에폭 수. (`10000`)
- **learning_rate**: 모델의 학습률. (`2e-4`)
- **betas**: Adam 옵티마이저의 beta 값. (`[0.8, 0.99]`)
- **eps**: Adam 옵티마이저의 epsilon 값. (`1e-9`)
- **batch_size**: 학습할 때 사용할 미니배치의 크기. (`16`)
- **fp16_run**: 16-bit 혼합 정밀도를 사용할지 여부. (`true`)
- **lr_decay**: 학습률 감쇠율. (`0.999875`)
- **segment_size**: 음성 샘플의 세그먼트 크기. (`8192`)
- **init_lr_ratio**: 초기 학습률 비율. (`1`)
- **warmup_epochs**: 워밍업할 에폭 수. (`0`)
- **c_mel**: 멜 스펙트로그램 손실 계수. (`45`)
- **c_kl**: KL 다이버전스 손실 계수. (`1.0`)

## 2. 데이터 설정 (`data`)

- **training_files**: 학습 데이터의 경로. (`../CH_JA_EN_mix_voice/clipped_3_vits_trilingual_annotations.train.txt.cleaned`)
- **validation_files**: 검증 데이터의 경로. (`../CH_JA_EN_mix_voice/clipped_3_vits_trilingual_annotations.val.txt.cleaned`)
- **text_cleaners**: 텍스트 전처리를 위한 클리너 목록. (`["cjke_cleaners2"]`)
- **max_wav_value**: 오디오 파일의 최대 진폭 값. (`32768.0`)
- **sampling_rate**: 오디오의 샘플링 레이트. (`22050`)
- **filter_length**: STFT 필터 길이. (`1024`)
- **hop_length**: STFT hop 길이. (`256`)
- **win_length**: STFT 윈도우 길이. (`1024`)
- **n_mel_channels**: 멜 스펙트로그램 채널 수. (`80`)
- **mel_fmin**: 멜 스케일에서의 최소 주파수. (`0.0`)
- **mel_fmax**: 멜 스케일에서의 최대 주파수. (`null` - 제한 없음)
- **add_blank**: 음소 간 블랭크 추가 여부. (`true`)
- **n_speakers**: 스피커 수. (`999`)
- **cleaned_text**: 텍스트가 사전 전처리되었는지 여부. (`true`)

## 3. 모델 설정 (`model`)

- **inter_channels**: 중간 채널 수. (`192`)
- **hidden_channels**: 은닉층 채널 수. (`192`)
- **filter_channels**: 필터 채널 수. (`768`)
- **n_heads**: 어텐션 헤드 수. (`2`)
- **n_layers**: 트랜스포머 레이어 수. (`6`)
- **kernel_size**: 합성곱 커널 크기. (`3`)
- **p_dropout**: 드롭아웃 확률. (`0.1`)
- **resblock**: ResBlock의 유형. (`1`)
- **resblock_kernel_sizes**: ResBlock에서 사용되는 커널 크기 배열. (`[3,7,11]`)
- **resblock_dilation_sizes**: ResBlock에서 사용되는 팽창 계수 배열. (`[[1,3,5], [1,3,5], [1,3,5]]`)
- **upsample_rates**: 업샘플링 비율. (`[8,8,2,2]`)
- **upsample_initial_channel**: 업샘플링 초기 채널 수. (`512`)
- **upsample_kernel_sizes**: 업샘플링 커널 크기 배열. (`[16,16,4,4]`)
- **n_layers_q**: 쿼리 레이어 수. (`3`)
- **use_spectral_norm**: 스펙트럼 정규화 사용 여부. (`false`)
- **gin_channels**: GIN(Global Information Network) 채널 수. (`256`)

## 4. 심볼 설정 (`symbols`)

- 음성 합성에 사용되는 기호 목록. 각 기호는 음소, 특수 문자 등 텍스트를 처리할 때 사용됩니다.

## 5. 스피커 설정 (`speakers`)

- 학습에 사용되는 스피커 캐릭터 목록과 각 캐릭터에 부여된 고유 ID입니다. 캐릭터는 다양한 애니메이션 및 게임의 캐릭터들로 구성되어 있습니다.
  예를 들어:
  
  - "特别周 Special Week (Umamusume Pretty Derby)": 0
  - "无声铃鹿 Silence Suzuka (Umamusume Pretty Derby)": 1
  - "菲谢尔 Fishl (Genshin Impact)": 230




  ### finetune_speaker.json
```json
{
  "train": {  // 학습 설정
    "log_interval": 200,  // 로그를 출력할 학습 단계 간격
    "eval_interval": 1000,  // 평가를 수행할 단계 간격
    "seed": 1234,  // 랜덤 시드 설정
    "epochs": 10000,  // 학습할 총 에폭 수
    "learning_rate": 2e-4,  // 학습률 (2 x 10^-4)
    "betas": [0.8, 0.99],  // Adam 옵티마이저의 beta 파라미터
    "eps": 1e-9,  // Adam 옵티마이저의 epsilon 값 (숫자 안정성 보장)
    "batch_size": 16,  // 미니배치 크기
    "fp16_run": true,  // 혼합 정밀도(half precision, 16-bit) 사용 여부
    "lr_decay": 0.999875,  // 학습률 감쇠율
    "segment_size": 8192,  // 각 음성 샘플의 세그먼트 크기
    "init_lr_ratio": 1,  // 초기 학습률 비율
    "warmup_epochs": 0,  // 학습 초반에 워밍업할 에폭 수
    "c_mel": 45,  // 멜 스펙트로그램 손실 계수
    "c_kl": 1.0  // KL 다이버전스 손실 계수
  },
  "data": {  // 데이터 설정
    "training_files": "../CH_JA_EN_mix_voice/clipped_3_vits_trilingual_annotations.train.txt.cleaned",  // 학습 데이터 경로
    "validation_files": "../CH_JA_EN_mix_voice/clipped_3_vits_trilingual_annotations.val.txt.cleaned",  // 검증 데이터 경로
    "text_cleaners": ["cjke_cleaners2"],  // 텍스트 전처리 클리너 목록
    "max_wav_value": 32768.0,  // 오디오 신호의 최대 진폭 값
    "sampling_rate": 22050,  // 오디오의 샘플링 레이트 (22050Hz)
    "filter_length": 1024,  // STFT 필터 길이
    "hop_length": 256,  // STFT hop 길이
    "win_length": 1024,  // STFT 윈도우 길이
    "n_mel_channels": 80,  // 멜 스펙트로그램 채널 수
    "mel_fmin": 0.0,  // 멜 스케일에서 최소 주파수
    "mel_fmax": null,  // 멜 스케일에서 최대 주파수 (제한 없음)
    "add_blank": true,  // 음소 간 블랭크 추가 여부
    "n_speakers": 999,  // 스피커 수 (999명의 스피커)
    "cleaned_text": true  // 텍스트 전처리가 완료되었는지 여부
  },
  "model": {  // 모델 설정
    "inter_channels": 192,  // 중간 채널 수
    "hidden_channels": 192,  // 은닉층 채널 수
    "filter_channels": 768,  // 필터 채널 수
    "n_heads": 2,  // 어텐션 헤드 수
    "n_layers": 6,  // 트랜스포머 레이어 수
    "kernel_size": 3,  // 합성곱 커널 크기
    "p_dropout": 0.1,  // 드롭아웃 확률
    "resblock": "1",  // ResBlock의 유형
    "resblock_kernel_sizes": [3,7,11],  // ResBlock의 커널 크기 배열
    "resblock_dilation_sizes": [[1,3,5], [1,3,5], [1,3,5]],  // ResBlock에서 사용되는 팽창 계수 배열
    "upsample_rates": [8,8,2,2],  // 업샘플링 비율
    "upsample_initial_channel": 512,  // 업샘플링 초기 채널 수
    "upsample_kernel_sizes": [16,16,4,4],  // 업샘플링 커널 크기 배열
    "n_layers_q": 3,  // 쿼리 레이어 수
    "use_spectral_norm": false,  // 스펙트럼 정규화 사용 여부
    "gin_channels": 256  // GIN(Global Information Network) 채널 수
  },
  "symbols": [  // 음성 합성에 사용되는 심볼(문자 및 특수기호) 목록
    "_", ",", ".", "!", "?", "-", "~", "…", "N", "Q", "a", "b", "d", "e", "f", "g", "h", "i", "j", "k", 
    "l", "m", "n", "o", "p", "s", "t", "u", "v", "w", "x", "y", "z", "ɑ", "æ", "ʃ", "ʑ", "ç", "ʊ", "ɪ", 
    "ɔ", "ɛ", "ɹ", "ð", "ə", "ɫ", "ɥ", "θ", "β", "ŋ", "ɦ", "⁼", "ʰ", "`", "^", "#", "*", "=", "ˈ", "ˌ", 
    "→", "↓", "↑", " "
  ],
  "speakers": {  // 각 스피커(캐릭터) 목록과 고유 ID
    "特别周 Special Week (Umamusume Pretty Derby)": 0,
    "无声铃鹿 Silence Suzuka (Umamusume Pretty Derby)": 1,
    "东海帝王 Tokai Teio (Umamusume Pretty Derby)": 2,
    ...
    "菲谢尔 Fishl (Genshin Impact)": 230
  }
}
```