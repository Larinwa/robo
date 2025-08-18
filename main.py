import sys
import config
from advanced_rag import RAGPipeline
from stt import SpeechToText
from tts import TextToSpeech

def run_chat_interface(pipeline: RAGPipeline, stt: SpeechToText, tts: TextToSpeech):
    print("\n--- Roboteacher is ready ---")
    print("Type 'exit' or 'quit' to end the session.")
    
    # Choose input mode
    mode = ""
    while mode not in ["1", "2"]:
        mode = input("\nChoose input mode: 1 = Text, 2 = Voice\nYour choice: ")

    while True:
        try:
            if mode == "2":
                # Voice input
                question = stt.listen()
                print(f"\nYou said: {question}")
            else:
                # Text input
                question = input("\nAsk a question: ")

            if question.strip().lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            print("Thinking...")
            answer = pipeline.invoke(question)

            # Print answer
            print(f"\nRoboteisha: {answer}")

            # Speak answer if not empty
            if answer.strip():
                try:
                    tts.speak(answer, output_path="roboteacher_output.wav")
                except Exception as e:
                    print(f"[TTS Error] Could not speak the answer: {e}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            sys.exit(1)

def main():
    try:
        pipeline = RAGPipeline(config)
        stt = SpeechToText(model_path="vosk-model-small-en-us-0.15")
        tts = TextToSpeech(model_name="tts_models/en/ljspeech/tacotron2-DDC")
        run_chat_interface(pipeline, stt, tts)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure your source document is available at the path specified in config.py")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during initialization: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
