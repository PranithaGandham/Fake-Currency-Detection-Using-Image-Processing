import sounddevice as sd
import soundfile as sf

def play_audio(denomination):
    audio_file = f'Deployment/audio/{denomination}.mp3'
    data, samplerate = sf.read(audio_file)
    sd.play(data, samplerate)
    sd.wait()  # Wait until the audio finishes playing

def play_audio_a(a):
    if a:
        audio_file = f'Deployment/audio/real.mp3'
    else:
        audio_file = f'Deployment/audio/fake.mp3'

    data, samplerate = sf.read(audio_file)
    sd.play(data, samplerate)
    sd.wait()  # Wait until the audio finishes playing