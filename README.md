# VITS-fast-fine-tuning-MIN
VITS-fast-fine-tuning 레포를 내 니즈에 맞춰 번경 중.

### SHORT-TERM - TODO
- 모델을 파인튜닝하기 위한 데이터셋인 'sampled_audio4ft_v2.zip' 파일을 Hugging Face에서 다운로드합니다.
- !wget https://huggingface.co/datasets/Plachta/sampled_audio4ft/resolve/main/sampled_audio4ft_v2.zip
- 위에서 언급 된 pretrainmodel 더 적당한게 있는지. 한영일 3개국어 베이스로 찾아보기.

- STEP 1.5 사전 훈련 모델 선택 파트에서 n개국어 별로 받아오는 모델이 다르다. 각각 특성 분석해보고 한국어 모델은 조금 다르게 받아오는데 차이 분석 필요.


### LONGTERM - TODO
긴 동영상 or 소리를 정해둔 길이에 맞춰서 자르는 기능
사운드 시각화 하여 전처리에 사용
모델 파인튜닝 이후 infer 방법 연구.
1차적으로 디스코드 TTS 봇 적용.