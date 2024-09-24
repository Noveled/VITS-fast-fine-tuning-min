import os
import numpy as np
import torch
from torch import no_grad, LongTensor
import argparse
import commons
from mel_processing import spectrogram_torch
import utils
from models import SynthesizerTrn
import gradio as gr # Gradio 라이브러리를 사용하여 웹 UI 생성
import librosa # 오디오 처리 라이브러리
import webbrowser # 웹 브라우저를 열기 위한 라이브러리

from text import text_to_sequence, _clean_text # 텍스트를 시퀀스로 변환하는 함수들

device = "cuda:0" if torch.cuda.is_available() else "cpu" # GPU 사용 여부에 따라 장치 설정
import logging
# 여러 라이브러리의 불필요한 로그 메시지를 숨김
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)

# 언어별로 적용할 텍스트 마크 설정
language_marks = {
    "Japanese": "",
    "日本語": "[JA]", # 일본어
    "简体中文": "[ZH]", # 중국어
    "English": "[EN]", # 영어
    "Mix": "", # 혼합 언어
}
lang = ['日本語', '简体中文', 'English', 'Mix'] # 사용 가능한 언어 목록

# 텍스트를 모델 입력에 맞게 변환하는 함수
def get_text(text, hps, is_symbol):
    # 텍스트를 시퀀스로 변환
    text_norm = text_to_sequence(text, hps.symbols, [] if is_symbol else hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0) # 블랭크 추가 (음소 간)
    text_norm = LongTensor(text_norm) # 텐서로 변환
    return text_norm

# 텍스트를 음성으로 변환하는 함수 생성
def create_tts_fn(model, hps, speaker_ids):
    def tts_fn(text, speaker, language, speed): # 텍스트, 캐릭터, 언어, 속도
        if language is not None:
            # 언어 마크를 텍스트 앞뒤에 추가
            text = language_marks[language] + text + language_marks[language]
        speaker_id = speaker_ids[speaker] # 선택된 스피커 ID
        stn_tst = get_text(text, hps, False) # 텍스트를 모델 입력에 맞게 변환
        with no_grad(): # 그래디언트 계산 비활성화 (평가 모드)
            x_tst = stn_tst.unsqueeze(0).to(device) # 배치 차원 추가 후 GPU로 전송
            x_tst_lengths = LongTensor([stn_tst.size(0)]).to(device) # 텍스트 길이 정보
            sid = LongTensor([speaker_id]).to(device) # 스피커 ID 정보
            # 모델을 통해 음성 생성 (속도 조절 가능)
            audio = model.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8,
                                length_scale=1.0 / speed)[0][0, 0].data.cpu().float().numpy()
        del stn_tst, x_tst, x_tst_lengths, sid # 메모리 해제
        return "Success", (hps.data.sampling_rate, audio) # 성공 메시지 및 생성된 오디오 반환

    return tts_fn

# 음성 변환(Voice Conversion) 함수 생성
def create_vc_fn(model, hps, speaker_ids):
    def vc_fn(original_speaker, target_speaker, record_audio, upload_audio):
        # 입력 오디오(녹음 또는 업로드된 파일)를 가져옴
        input_audio = record_audio if record_audio is not None else upload_audio
        if input_audio is None: # 오디오가 없으면 에러 메시지 출력
            return "You need to record or upload an audio", None
        sampling_rate, audio = input_audio # 샘플링 레이트와 오디오 데이터
        original_speaker_id = speaker_ids[original_speaker] # 원본 스피커 ID
        target_speaker_id = speaker_ids[target_speaker] # 타겟 스피커 ID

        # 오디오 데이터 정규화 및 처리
        audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
        if len(audio.shape) > 1: # 다중 채널 오디오일 경우 모노로 변환
            audio = librosa.to_mono(audio.transpose(1, 0))
        if sampling_rate != hps.data.sampling_rate: # 샘플링 레이트가 모델과 다를 경우 리샘플링
            audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=hps.data.sampling_rate)
        with no_grad(): # 그래디언트 계산 비활성화 (평가 모드)
            y = torch.FloatTensor(audio) # 오디오 데이터를 텐서로 변환
            y = y / max(-y.min(), y.max()) / 0.99 # 정규화
            y = y.to(device) # GPU로 전송
            y = y.unsqueeze(0) # 배치 차원 추가
            # 오디오의 스펙트로그램 생성
            spec = spectrogram_torch(y, hps.data.filter_length,
                                     hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                                     center=False).to(device)
            spec_lengths = LongTensor([spec.size(-1)]).to(device) # 스펙트로그램 길이 정보
            sid_src = LongTensor([original_speaker_id]).to(device) # 원본 스피커 ID
            sid_tgt = LongTensor([target_speaker_id]).to(device) # 타겟 스피커 ID
            # 음성 변환(Voice Conversion) 수행
            audio = model.voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt)[0][
                0, 0].data.cpu().float().numpy() # 변환된 오디오 반환
        del y, spec, spec_lengths, sid_src, sid_tgt # 메모리 해제
        return "Success", (hps.data.sampling_rate, audio) # 성공 메시지 및 변환된 오디오 반환

    return vc_fn
if __name__ == "__main__":
    # 명령줄 인자 설정
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="./G_latest.pth", help="directory to your fine-tuned model")
    parser.add_argument("--config_dir", default="./finetune_speaker.json", help="directory to your model config file")
    parser.add_argument("--share", default=False, help="make link public (used in colab)")

    args = parser.parse_args()
    hps = utils.get_hparams_from_file(args.config_dir) # 설정 파일 로드

    # 미세조정된 음성 합성 모델(SynthesizerTrn) 로드
    net_g = SynthesizerTrn(
        len(hps.symbols), # 심볼의 개수
        hps.data.filter_length // 2 + 1, # 필터 길이를 바탕으로 주파수 축 크기 설정
        hps.train.segment_size // hps.data.hop_length, # 세그먼트 크기 설정
        n_speakers=hps.data.n_speakers, # 스피커 수
        **hps.model).to(device) # 모델을 GPU로 전송
    _ = net_g.eval() # 평가 모드로 설정

    # 미세조정된 모델 체크포인트 로드
    _ = utils.load_checkpoint(args.model_dir, net_g, None)
    speaker_ids = hps.speakers # 스피커 ID 목록 로드
    speakers = list(hps.speakers.keys()) # 스피커 이름 목록 생성
    tts_fn = create_tts_fn(net_g, hps, speaker_ids) # 텍스트를 음성으로 변환하는 함수 생성
    vc_fn = create_vc_fn(net_g, hps, speaker_ids) # 음성 변환 함수 생성

    # Gradio UI 설정
    app = gr.Blocks()
    with app:
        with gr.Tab("Text-to-Speech"): # TTS 탭
            with gr.Row(): # 그리드 레이아웃에서 행(Row)을 생성
                with gr.Column(): # 첫 번째 열에 TTS 입력을 위한 컴포넌트들 배치
                    # 텍스트 입력란 (사용자가 변환하고자 하는 텍스트를 입력)
                    textbox = gr.TextArea(label="Text", # 입력란에 표시될 레이블
                                          placeholder="Type your sentence here", # 텍스트 입력 전 기본 힌트 텍스트
                                          value="こんにちわ。", # 기본 값으로 설정된 텍스트
                                          elem_id=f"tts-input") # HTML 요소의 ID (CSS 스타일링 등을 위해 사용)
                    # select character
                    # 캐릭터 선택 드롭다운
                    char_dropdown = gr.Dropdown(choices=speakers, # 스피커 목록에서 선택 가능 (여러 캐릭터 중 선택)
                                                value=speakers[0], # 기본 선택된 캐릭터 (첫 번째 스피커)
                                                label='character') # 드롭다운 메뉴의 레이블
                    
                    # 언어 선택 드롭다운 (일본어, 중국어, 영어 등 선택 가능)
                    language_dropdown = gr.Dropdown(choices=lang, # 사용 가능한 언어 목록
                                                    value=lang[0], # 기본 언어 (첫 번째 언어)
                                                    label='language') # 드롭다운 메뉴의 레이블
                    # 음성 생성 속도를 조절하는 슬라이더
                    duration_slider = gr.Slider(minimum=0.1, # 최소 속도
                                                maximum=5, # 최대 속도
                                                value=1, # 기본 설정된 속도 값 (1배속)
                                                step=0.1, # 슬라이더의 변화 단위 (0.1 단위로 속도 조절 가능)
                                                label='速度 Speed') # 슬라이더 레이블 (속도)
                with gr.Column(): # 두 번째 열에 TTS 출력 및 버튼 배치
                    # 메시지를 표시하는 텍스트 박스 (생성된 결과 상태 메시지 표시)
                    text_output = gr.Textbox(label="Message")
                    # 생성된 오디오를 재생할 수 있는 오디오 플레이어
                    audio_output = gr.Audio(label="Output Audio", elem_id="tts-audio")
                    # 음성 생성을 실행하는 버튼
                    btn = gr.Button("Generate!") # 버튼에 표시될 텍스트 ("Generate!")
                    # 버튼 클릭 시 TTS 함수(tts_fn)를 호출하여 음성을 생성
                    btn.click(tts_fn, # 버튼 클릭 시 호출할 함수 (음성 생성 함수)
                              inputs=[textbox, char_dropdown, language_dropdown, duration_slider,], # 입력으로 받을 값들 (텍스트, 캐릭터, 언어, 속도)
                              outputs=[text_output, audio_output]) # 함수 실행 후 출력할 값들 (메시지, 생성된 오디오)
        with gr.Tab("Voice Conversion"): # 음성 변환 탭
            gr.Markdown("""
                            录制或上传声音，并选择要转换的音色。
            """)
            with gr.Column():
                record_audio = gr.Audio(label="record your voice", source="microphone")
                upload_audio = gr.Audio(label="or upload audio here", source="upload")
                source_speaker = gr.Dropdown(choices=speakers, value=speakers[0], label="source speaker")
                target_speaker = gr.Dropdown(choices=speakers, value=speakers[0], label="target speaker")
            with gr.Column():
                message_box = gr.Textbox(label="Message")
                converted_audio = gr.Audio(label='converted audio')
            btn = gr.Button("Convert!")
            btn.click(vc_fn, inputs=[source_speaker, target_speaker, record_audio, upload_audio],
                      outputs=[message_box, converted_audio])
    webbrowser.open("http://127.0.0.1:7860") # 로컬 호스트에서 Gradio 앱 실행
    app.launch(share=args.share) # 앱 실행 및 공유 여부 설정

