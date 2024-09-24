import scipy.io.wavfile as wavfile

# 함수에서 반환된 값
sampling_rate, audio_data = hps.data.sampling_rate, audio

# 오디오 데이터를 .wav 파일로 저장
wavfile.write("output_audio.wav", sampling_rate, audio_data)