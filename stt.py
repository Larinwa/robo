import os
import queue
import sounddevice as sd
import vosk
import sys
import json
import select
from pathlib import Path

class SpeechToText:
    def __init__(self, model_path: str = "vosk-model-small-en-us-0.15"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Vosk model not found at {model_path}. Please download and extract it.")
        self.model = vosk.Model(model_path)
        self.q = queue.Queue()
        self.samplerate = 16000  # standard for Vosk

    def callback(self, indata, frames, time, status):
        """Called from a separate thread for each audio block."""
        if status:
            print(status, file=sys.stderr)
        self.q.put(bytes(indata))

    def listen(self) -> str:
        """
        Records audio until the user presses Enter and returns the full transcribed text.
        """
        rec = vosk.KaldiRecognizer(self.model, self.samplerate)
        print("Start speaking. Press Enter when done...")

        text = ""
        with sd.RawInputStream(
            samplerate=self.samplerate,
            blocksize=8000,
            dtype='int16',
            channels=1,
            callback=self.callback
        ):
            try:
                while True:
                    # Stop when Enter is pressed
                    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                        input()  # consume Enter
                        break
                    data = self.q.get()
                    if rec.AcceptWaveform(data):
                        result_json = json.loads(rec.Result())
                        text += result_json.get("text", "") + " "
            except KeyboardInterrupt:
                pass

            # Append final transcription
            final_result = json.loads(rec.FinalResult())
            text += final_result.get("text", "")

        return text.strip()


# -------------------------------
# Test block to run STT standalone
# -------------------------------
if __name__ == "__main__":
    ROOT_DIR = Path(__file__).parent
    MODEL_PATH = ROOT_DIR / "vosk-model-small-en-us-0.15"

    stt = SpeechToText(model_path=str(MODEL_PATH))
    text = stt.listen()
    print("Transcribed text:", text)
