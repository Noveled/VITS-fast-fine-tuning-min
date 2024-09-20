# Modified Finetune Speaker Configuration

이 문서는 `modified_finetune_speaker.json` 파일의 각 항목에 대한 설명을 제공합니다. 이 파일은 음성 합성 모델을 미세 조정하기 위한 설정 정보를 포함하고 있습니다.

## 1. 학습 설정 (`train`)

- **log_interval**: 학습 로그를 출력할 학습 단계 간격. (`10`)
- **eval_interval**: 검증을 수행할 단계 간격. (`100`)
- **seed**: 랜덤 시드 값, 재현성을 보장하기 위해 사용됨. (`1234`)
- **epochs**: 학습할 총 에포크 수. (`10000`)
- **learning_rate**: 학습률. (`0.0002`)
- **betas**: Adam 옵티마이저의 모멘텀을 설정하는 beta 값. (`[0.8, 0.99]`)
- **eps**: Adam 옵티마이저의 epsilon 값, 숫자 안정성을 보장. (`1e-09`)
- **batch_size**: 미니배치 크기, 한 번에 처리할 샘플의 수. (`16`)
- **fp16_run**: 혼합 정밀도(16-bit) 사용 여부, 메모리 절약 및 학습 속도 향상을 위해 사용됨. (`true`)
- **lr_decay**: 학습률 감쇠율, 학습이 진행됨에 따라 학습률을 감소시킴. (`0.999875`)
- **segment_size**: 각 음성 샘플의 세그먼트 크기. (`8192`)
- **init_lr_ratio**: 초기 학습률 비율. (`1`)
- **warmup_epochs**: 학습 초기 워밍업할 에포크 수. (`0`)
- **c_mel**: 멜 스펙트로그램 손실 계수. (`45`)
- **c_kl**: KL 다이버전스 손실 계수. (`1.0`)

## 2. 데이터 설정 (`data`)

- **training_files**: 학습 데이터 파일 경로. (`final_annotation_train.txt`)
- **validation_files**: 검증 데이터 파일 경로. (`final_annotation_val.txt`)
- **text_cleaners**: 텍스트 전처리를 위한 클리너 목록. (`["chinese_cleaners"]`)
- **max_wav_value**: 오디오 신호의 최대 진폭 값. (`32768.0`)
- **sampling_rate**: 오디오의 샘플링 레이트, 초당 샘플링 횟수. (`22050`)
- **filter_length**: STFT 필터 길이, 주파수 변환에 사용됨. (`1024`)
- **hop_length**: STFT hop 길이, 윈도우가 이동하는 간격. (`256`)
- **win_length**: STFT 윈도우 길이, 각 윈도우의 크기. (`1024`)
- **n_mel_channels**: 멜 스펙트로그램 채널 수. (`80`)
- **mel_fmin**: 멜 스케일에서의 최소 주파수. (`0.0`)
- **mel_fmax**: 멜 스케일에서의 최대 주파수. (`null`, 제한 없음)
- **add_blank**: 음소 간 공백 추가 여부. (`true`)
- **n_speakers**: 스피커 수. (`2`)
- **cleaned_text**: 텍스트가 전처리된 상태인지 여부. (`true`)

## 3. 모델 설정 (`model`)

- **inter_channels**: 중간 채널 수. (`192`)
- **hidden_channels**: 은닉층 채널 수. (`192`)
- **filter_channels**: 필터 채널 수. (`768`)
- **n_heads**: 어텐션 헤드 수. (`2`)
- **n_layers**: 트랜스포머 레이어 수. (`6`)
- **kernel_size**: 합성곱 커널 크기. (`3`)
- **p_dropout**: 드롭아웃 확률. (`0.1`)
- **resblock**: ResBlock의 유형. (`1`)
- **resblock_kernel_sizes**: ResBlock에서 사용하는 커널 크기 배열. (`[3, 7, 11]`)
- **resblock_dilation_sizes**: ResBlock에서 사용하는 팽창 계수 배열. (`[[1, 3, 5], [1, 3, 5], [1, 3, 5]]`)
- **upsample_rates**: 업샘플링 비율 배열. (`[8, 8, 2, 2]`)
- **upsample_initial_channel**: 업샘플링 초기 채널 수. (`512`)
- **upsample_kernel_sizes**: 업샘플링 커널 크기 배열. (`[16, 16, 4, 4]`)
- **n_layers_q**: 쿼리 레이어 수. (`3`)
- **use_spectral_norm**: 스펙트럼 정규화 사용 여부. (`false`)
- **gin_channels**: GIN(Global Information Network) 채널 수. (`256`)

## 4. 심볼 설정 (`symbols`)

음성 합성에 사용되는 기호 목록입니다. 다양한 특수 기호 및 알파벳을 포함하며, 중국어 및 영어 텍스트를 처리하기 위한 기호들입니다.

- **주요 기호**: `_`, `;`, `:`, `,`, `.`, `!`, `?`, `-`, `"`, `'`, `[]`, `()`, `...`, `—`
- **알파벳**: 대문자 A-Z, 소문자 a-z
- **숫자**: 0-5

## 5. 스피커 설정 (`speakers`)

각 스피커에 대한 이름과 고유 ID를 설정합니다. 여기서 스피커는 음성 합성에 사용될 캐릭터를 나타냅니다.

- **dingzhen**: 스피커 ID는 0
- **taffy**: 스피커 ID는 1




```json
{
  "train": {  // 학습 설정
    "log_interval": 10,  // 로그를 출력할 학습 단계 간격 (10단계마다 로그 출력)
    "eval_interval": 100,  // 검증을 수행할 단계 간격 (100단계마다 검증 수행)
    "seed": 1234,  // 랜덤 시드 값 (재현성 보장)
    "epochs": 10000,  // 학습할 총 에폭 수 (10000번 반복 학습)
    "learning_rate": 0.0002,  // 학습률 (2e-4)
    "betas": [  // Adam 옵티마이저의 beta 값 (모멘텀 설정)
      0.8,
      0.99
    ],
    "eps": 1e-09,  // Adam 옵티마이저의 epsilon 값 (숫자 안정성 보장)
    "batch_size": 16,  // 미니배치 크기 (한 번에 처리할 샘플 수)
    "fp16_run": true,  // 16-bit 혼합 정밀도 사용 여부 (메모리 절약 및 학습 속도 향상)
    "lr_decay": 0.999875,  // 학습률 감쇠율 (학습이 진행됨에 따라 학습률 감소)
    "segment_size": 8192,  // 각 음성 샘플의 세그먼트 크기 (단위: 샘플)
    "init_lr_ratio": 1,  // 초기 학습률 비율
    "warmup_epochs": 0,  // 학습 초기 워밍업할 에폭 수 (0이면 워밍업 없음)
    "c_mel": 45,  // 멜 스펙트로그램 손실 계수
    "c_kl": 1.0  // KL 다이버전스 손실 계수
  },
  "data": {  // 데이터 설정
    "training_files": "final_annotation_train.txt",  // 학습 데이터 파일 경로
    "validation_files": "final_annotation_val.txt",  // 검증 데이터 파일 경로
    "text_cleaners": [  // 텍스트 전처리에 사용할 클리너 목록
      "chinese_cleaners"
    ],
    "max_wav_value": 32768.0,  // 오디오 신호의 최대 진폭 값 (정규화된 값)
    "sampling_rate": 22050,  // 오디오의 샘플링 레이트 (22050Hz)
    "filter_length": 1024,  // STFT 필터 길이 (단위: 샘플)
    "hop_length": 256,  // STFT hop 길이 (단위: 샘플)
    "win_length": 1024,  // STFT 윈도우 길이 (단위: 샘플)
    "n_mel_channels": 80,  // 멜 스펙트로그램 채널 수 (80개 채널)
    "mel_fmin": 0.0,  // 멜 스케일에서의 최소 주파수 (0.0 Hz)
    "mel_fmax": null,  // 멜 스케일에서의 최대 주파수 (null은 제한 없음)
    "add_blank": true,  // 음소 간 공백 추가 여부
    "n_speakers": 2,  // 스피커 수 (2명의 스피커: dingzhen, taffy)
    "cleaned_text": true  // 텍스트가 전처리된 상태인지 여부 (true는 전처리 완료된 텍스트 사용)
  },
  "model": {  // 모델 설정
    "inter_channels": 192,  // 중간 채널 수
    "hidden_channels": 192,  // 은닉층 채널 수
    "filter_channels": 768,  // 필터 채널 수
    "n_heads": 2,  // 어텐션 헤드 수
    "n_layers": 6,  // 트랜스포머 레이어 수
    "kernel_size": 3,  // 합성곱 커널 크기
    "p_dropout": 0.1,  // 드롭아웃 확률 (0.1 = 10%)
    "resblock": "1",  // ResBlock의 유형 (1번 유형 사용)
    "resblock_kernel_sizes": [  // ResBlock에서 사용하는 커널 크기 배열
      3,
      7,
      11
    ],
    "resblock_dilation_sizes": [  // ResBlock에서 사용하는 팽창 계수 배열
      [1, 3, 5],
      [1, 3, 5],
      [1, 3, 5]
    ],
    "upsample_rates": [8, 8, 2, 2],  // 업샘플링 비율
    "upsample_initial_channel": 512,  // 업샘플링 시작 시의 채널 수
    "upsample_kernel_sizes": [16, 16, 4, 4],  // 업샘플링 커널 크기 배열
    "n_layers_q": 3,  // 쿼리 레이어 수
    "use_spectral_norm": false,  // 스펙트럼 정규화 사용 여부 (사용 안함)
    "gin_channels": 256  // GIN(Global Information Network) 채널 수
  },
  "symbols": [  // 음성 합성에 사용할 기호 목록 (중국어 및 영어 특수 문자 포함)
    "_",  // 공백
    "\uff1b",  // 세미콜론
    "\uff1a",  // 콜론
    "\uff0c",  // 쉼표
    "\u3002",  // 마침표
    "\uff01",  // 느낌표
    "\uff1f",  // 물음표
    "-",  // 하이픈
    "\u201c",  // 왼쪽 따옴표
    "\u201d",  // 오른쪽 따옴표
    "\u300a",  // 왼쪽 큰따옴표
    "\u300b",  // 오른쪽 큰따옴표
    "\u3001",  // 가운뎃점
    "\uff08",  // 왼쪽 괄호
    "\uff09",  // 오른쪽 괄호
    "\u2026",  // 줄임표
    "\u2014",  // 대시
    " ",  // 공백
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",  // 대문자 알파벳
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",  // 소문자 알파벳
    "1", "2", "3", "4", "5", "0",  // 숫자
    "\uff22",  // 풀와이드 'B'
    "\uff30"  // 풀와이드 'P'
  ],
  "speakers": {  // 스피커 정보 (각 스피커의 이름과 고유 ID)
    "dingzhen": 0,  // 'dingzhen' 스피커의 ID는 0
    "taffy": 1  // 'taffy' 스피커의 ID는 1
  }
}

```