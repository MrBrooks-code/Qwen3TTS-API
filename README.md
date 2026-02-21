# Qwen3-TTS Server

A Gradio web UI and REST API server for [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS), Alibaba's multilingual text-to-speech system.

Supports four TTS modes:
- **Custom Voice** — 9 built-in premium speakers with emotion/style control via natural language instructions
- **Voice Design** — describe a voice in plain English and the model creates it
- **Voice Cloning** — clone any voice from a ~3-second audio sample
- **Saved Voices** — save cloned voices by name and reuse them without re-uploading

## Requirements

- Python 3.9+
- ~10 GB disk space for model weights (downloaded automatically on first run)
- **Apple Silicon** — uses MPS (Metal) automatically
- **NVIDIA GPU** — uses CUDA + bfloat16 automatically
- **CPU** — works everywhere, but slow

## Setup

```bash
# Create and activate a conda environment
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Start the server

```bash
python app.py
```

This loads all three model variants and starts the server on `http://localhost:7860`. The app auto-detects the best available device: CUDA > MPS (Apple Silicon) > CPU.

**Options:**

| Flag        | Default  | Description |
|-------------|----------|-------------|
| `--device`  | `auto`   | Torch device: `auto`, `cuda:0`, `mps`, `cpu`. |
| `--dtype`   | `auto`   | Model precision: `auto`, `bf16`, `fp16`, `fp32`. Auto picks bf16 on CUDA, fp32 on MPS/CPU. |
| `--host`    | `0.0.0.0`| Server bind address. |
| `--port`    | `7860`   | Server port. |
| `--share`   | off      | Create a public Gradio share link. |

**Examples:**

```bash
# Auto-detect device (CUDA > MPS > CPU)
python app.py

# Force Apple Silicon GPU
python app.py --device mps

# Custom port
python app.py --port 8000

# Force CPU
python app.py --device cpu --dtype fp32

# Public share link
python app.py --share
```

### Web UI

Open `http://localhost:7860` in a browser. The interface has four tabs: Custom Voice, Voice Design, Voice Cloning, and Saved Voices.

### REST API

Full API documentation: [API.md](API.md)

Interactive Swagger docs: `http://localhost:7860/docs`

**Quick examples:**

```bash
# Generate speech with a built-in voice
curl -X POST http://localhost:7860/api/custom-voice \
  -F "text=Hello, how are you?" \
  -F "speaker=Ryan" \
  --output hello.wav

# Generate with emotion control
curl -X POST http://localhost:7860/api/custom-voice \
  -F "text=I can't believe we won!" \
  -F "speaker=Aiden" \
  -F "instruct=Very excited and happy" \
  --output excited.wav

# Design a new voice from a description
curl -X POST http://localhost:7860/api/voice-design \
  -F "text=Welcome to the show." \
  -F "instruct=A deep, gravelly male voice like a movie trailer narrator" \
  --output narrator.wav

# Clone a voice from a reference clip
curl -X POST http://localhost:7860/api/voice-clone \
  -F "text=New text spoken in the cloned voice." \
  -F "ref_audio=@sample.wav" \
  -F "ref_text=What was said in the sample clip." \
  --output cloned.wav

# Save a cloned voice for reuse
curl -X POST http://localhost:7860/api/voices \
  -F "name=My Narrator" \
  -F "ref_audio=@sample.wav" \
  -F "ref_text=What was said in the sample clip."

# Generate with a saved voice (no file upload needed)
curl -X POST http://localhost:7860/api/voices/My_Narrator/generate \
  -F "text=Now I can reuse this voice anytime." \
  -F "instruct=Speak cheerfully" \
  --output saved_voice.wav
```

## API Endpoints

| Method | Endpoint                          | Description |
|--------|-----------------------------------|-------------|
| GET    | `/api/speakers`                   | List available built-in speakers. |
| GET    | `/api/languages`                  | List supported languages (10 total). |
| POST   | `/api/custom-voice`               | Generate speech with a built-in voice + optional style instruction. |
| POST   | `/api/voice-design`               | Generate speech with a voice created from a text description. |
| POST   | `/api/voice-clone`                | Clone a voice from an uploaded reference audio clip. |
| GET    | `/api/voices`                     | List all saved cloned voices. |
| POST   | `/api/voices`                     | Save a cloned voice for reuse. |
| POST   | `/api/voices/{name}/generate`     | Generate speech with a saved voice (no upload needed). |
| DELETE | `/api/voices/{name}`              | Delete a saved voice. |

All POST generation endpoints accept `multipart/form-data` and return a WAV audio file.

See [API.md](API.md) for full parameter documentation, examples, and LLM tool definitions.

## LLM Integration

The API is designed to be called by LLM agents as a tool. [API.md](API.md) includes ready-to-use JSON tool definitions for:

- `tts_custom_voice` — generate speech with a built-in voice
- `tts_voice_design` — generate speech with a described voice
- `tts_voice_clone` — clone a voice from a reference audio file
- `tts_saved_voice` — generate speech with a saved cloned voice (simplest, no upload)
- `tts_save_voice` — save a reference audio as a named voice

### Python example for LLM tool use

```python
import requests

def tts_saved_voice(text, voice_name, language="English", instruct=""):
    """Generate speech using a saved cloned voice."""
    response = requests.post(f"http://localhost:7860/api/voices/{voice_name}/generate", data={
        "text": text,
        "language": language,
        "instruct": instruct,
    })
    response.raise_for_status()
    output_path = "output.wav"
    with open(output_path, "wb") as f:
        f.write(response.content)
    return output_path
```

## Supported Languages

Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian

## Built-in Speakers

| Speaker   | Voice                                    | Best Language |
|-----------|------------------------------------------|---------------|
| Vivian    | Bright, slightly edgy young female       | Chinese       |
| Serena    | Warm, gentle young female                | Chinese       |
| Uncle_Fu  | Seasoned male, low mellow timbre         | Chinese       |
| Dylan     | Youthful male, clear and natural         | Chinese       |
| Eric      | Lively male, slightly husky              | Chinese       |
| Ryan      | Dynamic male, strong rhythmic drive      | English       |
| Aiden     | Sunny American male, clear midrange      | English       |
| Ono_Anna  | Playful Japanese female, light and nimble| Japanese      |
| Sohee     | Warm Korean female, rich emotion         | Korean        |

## Project Structure

```
qwen3-tts/
  app.py             # Main server (Gradio UI + FastAPI REST API)
  requirements.txt   # Python dependencies
  API.md             # Full API documentation with LLM tool definitions
  README.md          # This file
  voices/            # Saved cloned voices (created at runtime)
```

## Powered by

[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by Alibaba Cloud — Apache 2.0 license.
