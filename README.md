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
