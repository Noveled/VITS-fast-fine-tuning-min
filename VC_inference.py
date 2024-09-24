import os
import numpy as np
import torch
from torch import no_grad, LongTensor
import argparse
import commons  # 일반적인 유틸리티 모듈
from mel_processing import spectrogram_torch  # 멜 스펙트로그램 생성 함수
import utils  # 다양한 유틸리티 함수 모음
from models import SynthesizerTrn  # 텍스트를 음성으로 변환하는 모델
import gradio as gr  # 웹 기반 인터페이스를 위한 Gradio 모듈
import librosa  # 오디오 처리를 위한 라이브러리
import webbrowser  # 웹브라우저를 열기 위한 모듈

from text import text_to_sequence, _clean_text  # 텍스트를 처리하고 시퀀스로 변환하는 함수
device = "cuda:0" if torch.cuda.is_available() else "cpu"  # GPU가 있으면 CUDA 장치를 사용하고, 없으면 CPU 사용

# 로그 레벨 설정 (경고만 출력)
import logging
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)

# 언어 마크 설정
language_marks = {
    "Japanese": "",  # 일본어는 빈 문자열로 처리
    "日本語": "[JA]",  # 일본어
    "简体中文": "[ZH]",  # 중국어
    "English": "[EN]",  # 영어
    "한국어": "[KO]",  # 한국어
    "Mix": "",  # 혼합
}

# 언어 목록
lang = ['日本語', '简体中文', 'English', '한국어', 'Mix']

# 텍스트를 모델 입력 형식으로 변환하는 함수
def get_text(text, hps, is_symbol):
    text_norm = text_to_sequence(text, hps.symbols, [] if is_symbol else hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)  # 입력 사이에 빈 공간 추가
    text_norm = LongTensor(text_norm)  # LongTensor로 변환
    return text_norm

# 텍스트를 음성으로 변환하는 함수 생성
def create_tts_fn(model, hps, speaker_ids):
    def tts_fn(text, speaker, language, speed):
        # 선택된 언어를 텍스트 앞뒤에 추가
        if language is not None:
            text = language_marks[language] + text + language_marks[language]
        speaker_id = speaker_ids[speaker]  # 선택된 스피커 ID
        stn_tst = get_text(text, hps, False)  # 텍스트를 모델 입력 형식으로 변환
        with no_grad():  # 평가 모드에서 그래디언트 계산 비활성화
            x_tst = stn_tst.unsqueeze(0).to(device)  # 배치 차원 추가 후 GPU로 전송
            x_tst_lengths = LongTensor([stn_tst.size(0)]).to(device)  # 텍스트 길이 정보
            sid = LongTensor([speaker_id]).to(device)  # 스피커 ID 정보
            # 음성 생성
            audio = model.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8,
                                length_scale=1.0 / speed)[0][0, 0].data.cpu().float().numpy()
        del stn_tst, x_tst, x_tst_lengths, sid  # 메모리 해제
        return "Success", (hps.data.sampling_rate, audio)  # 생성된 오디오 반환

    return tts_fn

# 음성 변환 함수 생성
def create_vc_fn(model, hps, speaker_ids):
    def vc_fn(original_speaker, target_speaker, record_audio, upload_audio):
        # 녹음된 오디오 또는 업로드된 오디오 중 하나를 사용
        input_audio = record_audio if record_audio is not None else upload_audio
        if input_audio is None:
            return "You need to record or upload an audio", None
        sampling_rate, audio = input_audio
        original_speaker_id = speaker_ids[original_speaker]  # 원본 스피커 ID
        target_speaker_id = speaker_ids[target_speaker]  # 목표 스피커 ID

        # 오디오 정규화 및 변환
        audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.transpose(1, 0))  # 모노 오디오로 변환
        if sampling_rate != hps.data.sampling_rate:
            audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=hps.data.sampling_rate)  # 샘플링 레이트 변경
        with no_grad():  # 그래디언트 계산 비활성화
            y = torch.FloatTensor(audio)
            y = y / max(-y.min(), y.max()) / 0.99  # 오디오 신호 정규화
            y = y.to(device)
            y = y.unsqueeze(0)  # 배치 차원 추가
            # 멜 스펙트로그램 생성
            spec = spectrogram_torch(y, hps.data.filter_length,
                                     hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                                     center=False).to(device)
            spec_lengths = LongTensor([spec.size(-1)]).to(device)
            sid_src = LongTensor([original_speaker_id]).to(device)  # 원본 스피커 ID
            sid_tgt = LongTensor([target_speaker_id]).to(device)  # 목표 스피커 ID
            # 음성 변환
            audio = model.voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt)[0][
                0, 0].data.cpu().float().numpy()
        del y, spec, spec_lengths, sid_src, sid_tgt  # 메모리 해제
        return "Success", (hps.data.sampling_rate, audio)  # 변환된 오디오 반환

    return vc_fn

# 메인 함수
if __name__ == "__main__":
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="./G_latest.pth", help="directory to your fine-tuned model")
    parser.add_argument("--config_dir", default="./finetune_speaker.json", help="directory to your model config file")
    parser.add_argument("--share", default=False, help="make link public (used in colab)")

    args = parser.parse_args()
    hps = utils.get_hparams_from_file(args.config_dir)

    # 모델 로드 및 설정
    net_g = SynthesizerTrn(
        len(hps.symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).to(device)
    _ = net_g.eval()  # 평가 모드로 전환

    # 모델 체크포인트 로드
    _ = utils.load_checkpoint(args.model_dir, net_g, None)
    speaker_ids = hps.speakers  # 스피커 ID 로드
    speakers = list(hps.speakers.keys())  # 스피커 목록 생성

    # TTS 함수와 음성 변환 함수 생성
    tts_fn = create_tts_fn(net_g, hps, speaker_ids)
    vc_fn = create_vc_fn(net_g, hps, speaker_ids)

    # Gradio UI 설정
    app = gr.Blocks()
    with app:
        with gr.Tab("Text-to-Speech"):  # 텍스트 -> 음성 탭
            with gr.Row():
                with gr.Column():
                    # 텍스트 입력
                    textbox = gr.TextArea(label="Text", placeholder="Type your sentence here", value="こんにちわ。", elem_id=f"tts-input")
                    # 캐릭터 선택
                    char_dropdown = gr.Dropdown(choices=speakers, value=speakers[0], label='character')
                    # 언어 선택
                    language_dropdown = gr.Dropdown(choices=lang, value=lang[0], label='language')
                    # 속도 조절 슬라이더
                    duration_slider = gr.Slider(minimum=0.1, maximum=5, value=1, step=0.1, label='速度 Speed')
                with gr.Column():
                    # 메시지 출력 및 오디오 출력
                    text_output = gr.Textbox(label="Message")
                    audio_output = gr.Audio(label="Output Audio", elem_id="tts-audio")
                    # 생성 버튼
                    btn = gr.Button("Generate!")
                    btn.click(tts_fn, inputs=[textbox, char_dropdown, language_dropdown, duration_slider], outputs=[text_output, audio_output])
        
        with gr.Tab("Voice Conversion"):  # 음성 변환 탭
            gr.Markdown("""录制或上传声音，并选择要转换的音色。""")
            with gr.Column():
                # 음성 녹음 및 업로드 기능
                record_audio = gr.Audio(label="record your voice", source="microphone")
                upload_audio = gr.Audio(label="or upload audio here", source="upload")
                source_speaker = gr.Dropdown(choices=speakers, value=speakers[0], label="source speaker")
                target_speaker = gr.Dropdown(choices=speakers, value=speakers[0], label="target speaker")
            with gr.Column():
                # 메시지와 변환된 오디오 출력
                message_box = gr.Textbox(label="Message")
                converted_audio = gr.Audio(label='converted audio')
            # 변환 버튼
            btn = gr.Button("Convert!")
            btn.click(vc_fn, inputs=[source_speaker, target_speaker, record_audio, upload_audio], outputs=[message_box, converted_audio])
    
    # 로컬 호스트에서 Gradio 앱 실행
    webbrowser.open("http://127.0.0.1:7860")
    app.launch(share=args.share)  # 공유 옵션 설정
