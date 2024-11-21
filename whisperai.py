import argparse
import whisper

def transcribe_audio(file_path, output_path):
    model = whisper.load_model("large")  #   Choose from tiny, base, small, medium, large
    result = model.transcribe(file_path)
    
    # Save transcription to output file
    with open(output_path, "w") as f:
        f.write(result["text"])
    
    print(f"Transcription saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe an audio file using Whisper.")
    parser.add_argument("--input", required=True, help="Path to the input audio file (e.g., meeting_recording.wav)")
    parser.add_argument("--output", required=True, help="Path to the output text file (e.g., transcription.txt)")
    args = parser.parse_args()

    # Transcribe the audio file and save to output file
    transcribe_audio(args.input, args.output)

