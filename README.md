# Audio Transcriber with Whisper

This Python script allows you to transcribe audio files into text using [OpenAI's Whisper model](https://github.com/openai/whisper). It leverages GPU (CUDA) for faster processing when available and processes audio in manageable chunks, displaying real-time progress through a `tqdm` progress bar.

## Features

- **Device Support**: Transcription on both CPU and GPU (if available) for optimized performance.
- **Parallel Processing**: Splits the audio into chunks and processes them concurrently.
- **Silent Mode**: Suppresses unnecessary logs for a cleaner experience.
- **Customizable Models**: Choose from Whisper's various models (`tiny`, `base`, `small`, `medium`, `large`) based on your requirements.
- **Interactive Progress Bar**: Displays real-time progress during transcription.

---

## How It Works

1. **Audio Loading**:
   - The script loads an audio file using `pydub` and splits it into chunks of a specified duration (default: 30 seconds).
2. **Model Loading**:
   - Models are loaded for the selected devices (CPU and/or GPU).
3. **Chunk Processing**:
   - Chunks are processed in parallel, with each chunk transcribed individually.
4. **Result Compilation**:
   - All transcriptions are combined into a single output text file.

---

## Prerequisites

Before using this script, ensure you have the following:

- **Python**: Version 3.8 or newer.
- **FFmpeg**: Required for audio processing by `pydub`.
- **CUDA**: For GPU support, ensure CUDA is installed and available.
- **Dependencies**:
  ```bash
  pip install whisper pydub tqdm torch

## Basic Knowledge
This script assumes a basic understanding of the following:

Python: Familiarity with running Python scripts and installing dependencies.
Command Line: Experience using terminal or command prompt for executing commands.
Machine Learning Models: Basic knowledge of AI/ML models (helpful but not mandatory).
## Usage
## Command
 ```bash
 python transcriber.py --input <input_audio_file> --output <output_text_file> [--device <device_list>] [--model <model_name>]
```

## Arguments
- --input: Path to the input audio file (e.g., meeting_recording.wav).
- --output: Path to the output text file (e.g., transcription.txt).
- --device (optional): Devices to use for processing, e.g., cpu,cuda. Defaults to cpu,cuda.
- --model (optional): Whisper model to use (tiny, base, small, medium, large). Defaults to large.

## Example
Transcribe an audio file using both CPU and GPU with the large model:

```bash
python transcriber.py --input meeting.wav --output transcription.txt --device cpu,cuda --model large
```

## Technologies Used
Whisper: State-of-the-art transcription model.
- PyDub: Audio processing library.
-  TQDM: Real-time progress bar visualization.
-   Torch: For GPU (CUDA) processing support.

## Script Highlights
### Silent Mode
The Silent class suppresses logs for a cleaner output:

```python
class Silent:
    def __init__(self, allow_tqdm=False):
        # Initializes silent mode
```

### Parallel Processing
Threads handle transcription on multiple devices concurrently:

```python

threads.append(threading.Thread(target=process_chunks, args=(indices, device), daemon=True))
```
Chunk Management
Audio files are split into manageable durations for processing:

```python
chunk = audio[start_ms:end_ms]
chunk.export(chunk_file, format="wav")
```
## Known Limitations
### Long Audio Files:
Very long files may require significant memory for processing.
### CUDA Availability:

Ensure `torch.cuda.is_available()` returns True for GPU usage.
