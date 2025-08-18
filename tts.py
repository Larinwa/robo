# tts.py
from TTS.api import TTS
import sounddevice as sd
import soundfile as sf
import os

class TextToSpeech:
    def __init__(self, model_name: str = "tts_models/en/ljspeech/tacotron2-DDC"):
        """
        Initialize TTS engine with a pre-trained model.
        """
        self.tts = TTS(model_name)

    def speak(self, text: str, output_path: str = "output.wav"):
        """
        Convert text to speech, save as a file, and play immediately.
        """
        # Generate audio file
        self.tts.tts_to_file(text=text, file_path=output_path)
        print(f"Audio saved to {output_path}")

        # Play the audio
        if os.path.exists(output_path):
            data, fs = sf.read(output_path, dtype='float32')
            sd.play(data, fs)
            sd.wait()  # Wait until playback is finished
        else:
            print("Error: Audio file not found.")
