"""
Qwen3-TTS Interface — Gradio UI + REST API

Supports all three modes:
  1. Custom Voice — 9 built-in speakers with emotion/instruction control
  2. Voice Design — create new voices from natural language descriptions
  3. Voice Cloning — clone any voice from a short reference audio clip
  4. Saved Voices — reuse previously cloned voices by name

Launch:
    python app.py                          # auto-detects best device (CUDA > MPS > CPU)
    python app.py --device cpu             # force CPU inference
    python app.py --device mps             # force Apple Silicon GPU
    python app.py --port 7860 --share      # custom port + public link

API endpoints (documented at /docs):
    POST /api/custom-voice
    POST /api/voice-design
    POST /api/voice-clone
    POST /api/voices                       # save a cloned voice
    GET  /api/voices                       # list saved voices
    POST /api/voices/{name}/generate       # generate with a saved voice
    DELETE /api/voices/{name}              # delete a saved voice
"""

import argparse
import json
import logging
import os
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from qwen_tts import Qwen3TTSModel

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="\033[90m%(asctime)s\033[0m %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("qwen3-tts")


def log_request(endpoint: str, params: dict):
    """Log an incoming API request with all parameters."""
    parts = [f"\033[1;36mAPI\033[0m \033[1m{endpoint}\033[0m"]
    for k, v in params.items():
        if v is None or v == "":
            continue
        display = v
        if isinstance(v, str) and len(v) > 80:
            display = v[:77] + "..."
        parts.append(f"  \033[33m{k}\033[0m: {display}")
    log.info("\n".join(parts))


def log_result(endpoint: str, duration: float, output_path: str = None, error: str = None):
    """Log the result of an API call."""
    if error:
        log.info(f"\033[1;31mERROR\033[0m {endpoint} failed after {duration:.1f}s: {error}")
    else:
        log.info(f"\033[1;32mDONE\033[0m {endpoint} \033[90m{duration:.1f}s\033[0m -> {output_path}")


# ---------------------------------------------------------------------------
# Globals (populated in main)
# ---------------------------------------------------------------------------
MODEL_CUSTOM_VOICE = None
MODEL_VOICE_DESIGN = None
MODEL_BASE = None
OUTPUT_DIR = Path(tempfile.mkdtemp(prefix="qwen3tts_"))
VOICES_DIR = Path(__file__).parent / "voices"

SPEAKERS = [
    "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric",
    "Ryan", "Aiden", "Ono_Anna", "Sohee",
]
LANGUAGES = [
    "Chinese", "English", "Japanese", "Korean",
    "German", "French", "Russian", "Portuguese", "Spanish", "Italian",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TARGET_SR = 44100


def save_wav(waveform: np.ndarray, sr: int, volume: float = 1.0) -> str:
    """Save a waveform as 44.1kHz 16-bit mono PCM WAV."""
    if volume != 1.0:
        waveform = np.clip(waveform * volume, -1.0, 1.0)

    # Ensure mono
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=-1)

    # Resample to 44.1kHz if needed
    if sr != TARGET_SR:
        import librosa
        waveform = librosa.resample(waveform.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)

    # Convert to 16-bit signed PCM
    pcm = np.clip(waveform, -1.0, 1.0)
    pcm = (pcm * 32767).astype(np.int16)

    path = OUTPUT_DIR / f"{uuid.uuid4().hex}.wav"
    sf.write(str(path), pcm, TARGET_SR, subtype="PCM_16")
    return str(path)


def sanitize_name(name: str) -> str:
    """Sanitize a voice name to a safe directory name."""
    return re.sub(r"[^\w\-]", "_", name.strip())


VALID_MODELS = {"custom", "design", "base"}


def load_models(device: str, dtype: torch.dtype, models: set[str]):
    """Load the requested model variants."""
    global MODEL_CUSTOM_VOICE, MODEL_VOICE_DESIGN, MODEL_BASE

    common = dict(device_map=device, dtype=dtype)

    if "custom" in models:
        log.info("Loading CustomVoice model...")
        MODEL_CUSTOM_VOICE = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", **common,
        )

    if "design" in models:
        log.info("Loading VoiceDesign model...")
        MODEL_VOICE_DESIGN = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", **common,
        )

    if "base" in models:
        log.info("Loading Base (voice clone / saved voices) model...")
        MODEL_BASE = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base", **common,
        )

    loaded = [m for m in ["custom", "design", "base"] if m in models]
    skipped = [m for m in ["custom", "design", "base"] if m not in models]
    log.info(f"Models loaded: {', '.join(loaded)}" + (f"  (skipped: {', '.join(skipped)})" if skipped else ""))


# ---------------------------------------------------------------------------
# Saved voices management
# ---------------------------------------------------------------------------

def list_saved_voices() -> list[dict]:
    """Return metadata for all saved voices."""
    voices = []
    if not VOICES_DIR.exists():
        return voices
    for voice_dir in sorted(VOICES_DIR.iterdir()):
        meta_path = voice_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            meta["id"] = voice_dir.name
            voices.append(meta)
    return voices


def get_saved_voice_names() -> list[str]:
    """Return a list of saved voice names for dropdowns."""
    return [v["name"] for v in list_saved_voices()]


def save_voice(name: str, ref_audio_path: str, ref_text: str, language: str) -> str:
    """Save a cloned voice (reference audio + metadata) to disk."""
    safe_name = sanitize_name(name)
    if not safe_name:
        raise ValueError("Voice name cannot be empty.")

    voice_dir = VOICES_DIR / safe_name
    voice_dir.mkdir(parents=True, exist_ok=True)

    # Copy reference audio
    audio_dest = voice_dir / "ref_audio.wav"
    data, sr = sf.read(ref_audio_path)
    sf.write(str(audio_dest), data, sr)

    # Save metadata
    meta = {
        "name": name,
        "ref_text": ref_text,
        "language": language,
    }
    with open(voice_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return safe_name


def delete_voice(name: str) -> bool:
    """Delete a saved voice by name."""
    safe_name = sanitize_name(name)
    voice_dir = VOICES_DIR / safe_name
    if voice_dir.exists():
        shutil.rmtree(voice_dir)
        return True
    return False


def get_voice_path_and_meta(name: str) -> tuple[str, dict]:
    """Get the ref audio path and metadata for a saved voice."""
    safe_name = sanitize_name(name)
    voice_dir = VOICES_DIR / safe_name
    meta_path = voice_dir / "meta.json"
    audio_path = voice_dir / "ref_audio.wav"
    if not meta_path.exists() or not audio_path.exists():
        raise FileNotFoundError(f"Saved voice '{name}' not found.")
    with open(meta_path) as f:
        meta = json.load(f)
    return str(audio_path), meta


# ---------------------------------------------------------------------------
# Core generation functions
# ---------------------------------------------------------------------------

def generate_custom_voice(
    text: str,
    language: str,
    speaker: str,
    instruct: str,
    volume: float,
    temperature: float,
    top_k: int,
    top_p: float,
):
    if MODEL_CUSTOM_VOICE is None:
        raise gr.Error("CustomVoice model not loaded. Start with --models custom or --models all.")
    if not text.strip():
        raise gr.Error("Please enter some text.")

    wavs, sr = MODEL_CUSTOM_VOICE.generate_custom_voice(
        text=text,
        language=language,
        speaker=speaker,
        instruct=instruct if instruct.strip() else None,
        temperature=temperature,
        top_k=int(top_k),
        top_p=top_p,
    )
    path = save_wav(wavs[0], sr, volume)
    return path


def generate_voice_design(
    text: str,
    language: str,
    instruct: str,
    volume: float,
    temperature: float,
    top_k: int,
    top_p: float,
):
    if MODEL_VOICE_DESIGN is None:
        raise gr.Error("VoiceDesign model not loaded. Start with --models design or --models all.")
    if not text.strip():
        raise gr.Error("Please enter some text.")
    if not instruct.strip():
        raise gr.Error("Please describe the voice you want.")

    wavs, sr = MODEL_VOICE_DESIGN.generate_voice_design(
        text=text,
        language=language,
        instruct=instruct,
        temperature=temperature,
        top_k=int(top_k),
        top_p=top_p,
    )
    path = save_wav(wavs[0], sr, volume)
    return path


def generate_voice_clone(
    text: str,
    language: str,
    ref_audio: str,
    ref_text: str,
    volume: float,
    temperature: float,
    top_k: int,
    top_p: float,
):
    if MODEL_BASE is None:
        raise gr.Error("Base model not loaded. Start with --models base or --models all.")
    if not text.strip():
        raise gr.Error("Please enter some text.")
    if ref_audio is None:
        raise gr.Error("Please upload a reference audio clip.")

    wavs, sr = MODEL_BASE.generate_voice_clone(
        text=text,
        language=language,
        ref_audio=ref_audio,
        ref_text=ref_text if ref_text.strip() else None,
        temperature=temperature,
        top_k=int(top_k),
        top_p=top_p,
    )
    path = save_wav(wavs[0], sr, volume)
    return path


def generate_from_saved_voice(
    text: str,
    language: str,
    voice_name: str,
    instruct: str,
    volume: float,
    temperature: float,
    top_k: int,
    top_p: float,
):
    if MODEL_BASE is None:
        raise gr.Error("Base model not loaded. Start with --models base or --models all.")
    if not text.strip():
        raise gr.Error("Please enter some text.")
    if not voice_name:
        raise gr.Error("Please select a saved voice.")

    try:
        audio_path, meta = get_voice_path_and_meta(voice_name)
    except FileNotFoundError:
        raise gr.Error(f"Saved voice '{voice_name}' not found.")

    ref_text = meta.get("ref_text") or None

    kwargs = dict(
        text=text,
        language=language,
        ref_audio=audio_path,
        ref_text=ref_text,
        temperature=temperature,
        top_k=int(top_k),
        top_p=top_p,
    )
    if instruct and instruct.strip():
        kwargs["instruct"] = instruct

    wavs, sr = MODEL_BASE.generate_voice_clone(**kwargs)
    path = save_wav(wavs[0], sr, volume)
    return path


# ---------------------------------------------------------------------------
# Gradio UI helpers
# ---------------------------------------------------------------------------

def ui_save_voice(name: str, ref_audio: str, ref_text: str, language: str):
    """Save a voice from the Gradio UI."""
    if not name.strip():
        raise gr.Error("Please enter a name for the voice.")
    if ref_audio is None:
        raise gr.Error("Please upload reference audio first.")
    save_voice(name, ref_audio, ref_text, language)
    new_choices = get_saved_voice_names()
    return (
        gr.Info(f"Voice '{name}' saved."),
        gr.update(choices=new_choices, value=name),
    )


def ui_delete_voice(voice_name: str):
    """Delete a voice from the Gradio UI."""
    if not voice_name:
        raise gr.Error("No voice selected.")
    delete_voice(voice_name)
    new_choices = get_saved_voice_names()
    return (
        gr.Info(f"Voice '{voice_name}' deleted."),
        gr.update(choices=new_choices, value=None),
    )


def ui_refresh_voices():
    """Refresh the saved voices dropdown."""
    return gr.update(choices=get_saved_voice_names())


def ui_load_voice_preview(voice_name: str):
    """Load the reference audio for preview when a saved voice is selected."""
    if not voice_name:
        return None
    try:
        audio_path, _ = get_voice_path_and_meta(voice_name)
        return audio_path
    except FileNotFoundError:
        return None


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_gradio_app() -> gr.Blocks:
    with gr.Blocks(title="Qwen3-TTS", theme=gr.themes.Soft()) as app:
        gr.Markdown("# Qwen3-TTS\nMultilingual text-to-speech with voice cloning, custom voices, and voice design.")

        with gr.Tab("Custom Voice"):
            gr.Markdown("Use one of the 9 built-in premium voices. Optionally add an instruction to control emotion, tone, and style.")
            with gr.Row():
                with gr.Column():
                    cv_text = gr.Textbox(label="Text", lines=4, placeholder="Enter the text to synthesize...")
                    with gr.Row():
                        cv_lang = gr.Dropdown(LANGUAGES, value="English", label="Language")
                        cv_speaker = gr.Dropdown(SPEAKERS, value="Ryan", label="Speaker")
                    cv_instruct = gr.Textbox(label="Instruction (optional)", placeholder="e.g. Speak warmly and cheerfully")
                    cv_vol = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Volume")
                    with gr.Accordion("Advanced", open=False):
                        cv_temp = gr.Slider(0.1, 2.0, value=0.9, step=0.05, label="Temperature")
                        cv_topk = gr.Slider(1, 200, value=50, step=1, label="Top-K")
                        cv_topp = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="Top-P")
                    cv_btn = gr.Button("Generate", variant="primary")
                with gr.Column():
                    cv_audio = gr.Audio(label="Output", type="filepath")
            cv_btn.click(
                generate_custom_voice,
                inputs=[cv_text, cv_lang, cv_speaker, cv_instruct, cv_vol, cv_temp, cv_topk, cv_topp],
                outputs=cv_audio,
            )

        with gr.Tab("Voice Design"):
            gr.Markdown("Describe the voice you want in natural language, and the model will synthesize speech with a matching voice.")
            with gr.Row():
                with gr.Column():
                    vd_text = gr.Textbox(label="Text", lines=4, placeholder="Enter the text to synthesize...")
                    vd_lang = gr.Dropdown(LANGUAGES, value="English", label="Language")
                    vd_instruct = gr.Textbox(
                        label="Voice Description",
                        lines=2,
                        placeholder="e.g. A warm young female voice with a slight edge",
                    )
                    vd_vol = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Volume")
                    with gr.Accordion("Advanced", open=False):
                        vd_temp = gr.Slider(0.1, 2.0, value=0.9, step=0.05, label="Temperature")
                        vd_topk = gr.Slider(1, 200, value=50, step=1, label="Top-K")
                        vd_topp = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="Top-P")
                    vd_btn = gr.Button("Generate", variant="primary")
                with gr.Column():
                    vd_audio = gr.Audio(label="Output", type="filepath")
            vd_btn.click(
                generate_voice_design,
                inputs=[vd_text, vd_lang, vd_instruct, vd_vol, vd_temp, vd_topk, vd_topp],
                outputs=vd_audio,
            )

        with gr.Tab("Voice Cloning"):
            gr.Markdown("Upload a short reference audio clip (~3 seconds) and the model will clone that voice for new text. You can save the voice for reuse in the Saved Voices tab.")
            with gr.Row():
                with gr.Column():
                    vc_text = gr.Textbox(label="Text", lines=4, placeholder="Enter the text to synthesize...")
                    vc_lang = gr.Dropdown(LANGUAGES, value="English", label="Language")
                    vc_ref_audio = gr.Audio(label="Reference Audio", type="filepath")
                    vc_ref_text = gr.Textbox(
                        label="Reference Transcript (optional)",
                        placeholder="Transcript of the reference audio for better quality",
                    )
                    vc_vol = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Volume")
                    with gr.Accordion("Advanced", open=False):
                        vc_temp = gr.Slider(0.1, 2.0, value=0.9, step=0.05, label="Temperature")
                        vc_topk = gr.Slider(1, 200, value=50, step=1, label="Top-K")
                        vc_topp = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="Top-P")
                    with gr.Row():
                        vc_btn = gr.Button("Generate", variant="primary")
                with gr.Column():
                    vc_audio = gr.Audio(label="Output", type="filepath")
                    gr.Markdown("### Save This Voice")
                    vc_save_name = gr.Textbox(label="Voice Name", placeholder="e.g. My Narrator")
                    vc_save_btn = gr.Button("Save Voice", variant="secondary")
                    vc_save_status = gr.Textbox(label="Status", interactive=False, visible=False)

            # Hidden dropdown to receive updates (keeps Saved Voices tab in sync)
            vc_hidden_dropdown = gr.Dropdown(visible=False)

            vc_btn.click(
                generate_voice_clone,
                inputs=[vc_text, vc_lang, vc_ref_audio, vc_ref_text, vc_vol, vc_temp, vc_topk, vc_topp],
                outputs=vc_audio,
            )
            vc_save_btn.click(
                ui_save_voice,
                inputs=[vc_save_name, vc_ref_audio, vc_ref_text, vc_lang],
                outputs=[vc_save_status, vc_hidden_dropdown],
            )

        with gr.Tab("Saved Voices"):
            gr.Markdown("Generate speech using a previously saved cloned voice.")
            with gr.Row():
                with gr.Column():
                    sv_voice = gr.Dropdown(
                        choices=get_saved_voice_names(),
                        label="Saved Voice",
                        interactive=True,
                    )
                    with gr.Row():
                        sv_refresh_btn = gr.Button("Refresh List", size="sm")
                        sv_delete_btn = gr.Button("Delete Voice", variant="stop", size="sm")
                    sv_preview = gr.Audio(label="Reference Audio Preview", type="filepath", interactive=False)
                    sv_text = gr.Textbox(label="Text", lines=4, placeholder="Enter the text to synthesize...")
                    sv_lang = gr.Dropdown(LANGUAGES, value="English", label="Language")
                    sv_instruct = gr.Textbox(label="Instruction (optional)", placeholder="e.g. Speak warmly and cheerfully")
                    sv_vol = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Volume")
                    with gr.Accordion("Advanced", open=False):
                        sv_temp = gr.Slider(0.1, 2.0, value=0.9, step=0.05, label="Temperature")
                        sv_topk = gr.Slider(1, 200, value=50, step=1, label="Top-K")
                        sv_topp = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="Top-P")
                    sv_btn = gr.Button("Generate", variant="primary")
                with gr.Column():
                    sv_audio = gr.Audio(label="Output", type="filepath")

            sv_delete_status = gr.Textbox(visible=False)

            sv_voice.change(ui_load_voice_preview, inputs=[sv_voice], outputs=[sv_preview])
            sv_refresh_btn.click(ui_refresh_voices, outputs=[sv_voice])
            sv_delete_btn.click(ui_delete_voice, inputs=[sv_voice], outputs=[sv_delete_status, sv_voice])
            sv_btn.click(
                generate_from_saved_voice,
                inputs=[sv_text, sv_lang, sv_voice, sv_instruct, sv_vol, sv_temp, sv_topk, sv_topp],
                outputs=sv_audio,
            )

    return app


# ---------------------------------------------------------------------------
# FastAPI REST endpoints
# ---------------------------------------------------------------------------

def require_model(model, name: str):
    """Raise if a model isn't loaded."""
    if model is None:
        raise ValueError(f"{name} model not loaded. Start with --models {name.lower()} or --models all.")


def build_api(fast_app: FastAPI):
    """Mount REST endpoints onto the FastAPI app that Gradio creates."""

    @fast_app.post("/api/custom-voice")
    async def api_custom_voice(
        text: str = Form(...),
        language: str = Form("English"),
        speaker: str = Form("Ryan"),
        instruct: str = Form(""),
        volume: float = Form(1.0),
        temperature: float = Form(0.9),
        top_k: int = Form(50),
        top_p: float = Form(1.0),
    ):
        """Generate speech using a built-in premium voice."""
        log_request("POST /api/custom-voice", {
            "text": text, "language": language, "speaker": speaker,
            "instruct": instruct, "volume": volume, "temperature": temperature,
            "top_k": top_k, "top_p": top_p,
        })
        t0 = time.time()
        try:
            require_model(MODEL_CUSTOM_VOICE, "custom")
            wavs, sr = MODEL_CUSTOM_VOICE.generate_custom_voice(
                text=text,
                language=language,
                speaker=speaker,
                instruct=instruct if instruct.strip() else None,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            path = save_wav(wavs[0], sr, volume)
            log_result("POST /api/custom-voice", time.time() - t0, path)
            return FileResponse(path, media_type="audio/wav", filename="output.wav")
        except Exception as e:
            log_result("POST /api/custom-voice", time.time() - t0, error=str(e))
            return JSONResponse(status_code=500, content={"error": str(e)})

    @fast_app.post("/api/voice-design")
    async def api_voice_design(
        text: str = Form(...),
        language: str = Form("English"),
        instruct: str = Form(...),
        volume: float = Form(1.0),
        temperature: float = Form(0.9),
        top_k: int = Form(50),
        top_p: float = Form(1.0),
    ):
        """Generate speech with a designed voice from a natural language description."""
        log_request("POST /api/voice-design", {
            "text": text, "language": language, "instruct": instruct,
            "volume": volume, "temperature": temperature, "top_k": top_k, "top_p": top_p,
        })
        t0 = time.time()
        try:
            require_model(MODEL_VOICE_DESIGN, "design")
            wavs, sr = MODEL_VOICE_DESIGN.generate_voice_design(
                text=text,
                language=language,
                instruct=instruct,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            path = save_wav(wavs[0], sr, volume)
            log_result("POST /api/voice-design", time.time() - t0, path)
            return FileResponse(path, media_type="audio/wav", filename="output.wav")
        except Exception as e:
            log_result("POST /api/voice-design", time.time() - t0, error=str(e))
            return JSONResponse(status_code=500, content={"error": str(e)})

    @fast_app.post("/api/voice-clone")
    async def api_voice_clone(
        text: str = Form(...),
        language: str = Form("English"),
        ref_text: str = Form(""),
        ref_audio: UploadFile = File(...),
        volume: float = Form(1.0),
        temperature: float = Form(0.9),
        top_k: int = Form(50),
        top_p: float = Form(1.0),
    ):
        """Clone a voice from a reference audio clip and synthesize new text."""
        log_request("POST /api/voice-clone", {
            "text": text, "language": language,
            "ref_audio": ref_audio.filename, "ref_text": ref_text,
            "volume": volume, "temperature": temperature, "top_k": top_k, "top_p": top_p,
        })
        t0 = time.time()
        try:
            require_model(MODEL_BASE, "base")
            ref_path = OUTPUT_DIR / f"ref_{uuid.uuid4().hex}.wav"
            contents = await ref_audio.read()
            with open(ref_path, "wb") as f:
                f.write(contents)

            wavs, sr = MODEL_BASE.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=str(ref_path),
                ref_text=ref_text if ref_text.strip() else None,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            path = save_wav(wavs[0], sr, volume)
            log_result("POST /api/voice-clone", time.time() - t0, path)
            return FileResponse(path, media_type="audio/wav", filename="output.wav")
        except Exception as e:
            log_result("POST /api/voice-clone", time.time() - t0, error=str(e))
            return JSONResponse(status_code=500, content={"error": str(e)})

    # ----- Saved voices endpoints -----

    @fast_app.get("/api/voices")
    async def api_list_voices():
        """List all saved cloned voices.

        Returns:
            {"voices": [{"id": "...", "name": "...", "ref_text": "...", "language": "..."}, ...]}
        """
        voices = list_saved_voices()
        log.info(f"\033[1;36mAPI\033[0m \033[1mGET /api/voices\033[0m -> {len(voices)} saved voice(s)")
        return {"voices": voices}

    @fast_app.post("/api/voices")
    async def api_save_voice(
        name: str = Form(...),
        language: str = Form("English"),
        ref_text: str = Form(""),
        ref_audio: UploadFile = File(...),
    ):
        """Save a cloned voice for later reuse.

        Parameters:
            name: A human-readable name for this voice (e.g. "My Narrator").
            language: The language spoken in the reference audio.
            ref_text: Transcript of the reference audio (improves quality).
            ref_audio: Reference audio file (WAV, MP3, etc.) — ~3 seconds recommended.

        Returns:
            {"id": "...", "name": "...", "message": "Voice saved."}
        """
        log_request("POST /api/voices", {
            "name": name, "language": language, "ref_text": ref_text,
            "ref_audio": ref_audio.filename,
        })
        try:
            ref_path = OUTPUT_DIR / f"ref_{uuid.uuid4().hex}.wav"
            contents = await ref_audio.read()
            with open(ref_path, "wb") as f:
                f.write(contents)
            voice_id = save_voice(name, str(ref_path), ref_text, language)
            log.info(f"\033[1;32mDONE\033[0m POST /api/voices -> saved as '{voice_id}'")
            return {"id": voice_id, "name": name, "message": "Voice saved."}
        except Exception as e:
            log.info(f"\033[1;31mERROR\033[0m POST /api/voices: {e}")
            return JSONResponse(status_code=500, content={"error": str(e)})

    @fast_app.post("/api/voices/{voice_name}/generate")
    async def api_generate_saved_voice(
        voice_name: str,
        text: str = Form(...),
        language: str = Form("English"),
        instruct: str = Form(""),
        volume: float = Form(1.0),
        temperature: float = Form(0.9),
        top_k: int = Form(50),
        top_p: float = Form(1.0),
    ):
        """Generate speech using a previously saved cloned voice.

        Parameters:
            voice_name: The name of the saved voice (as returned by GET /api/voices).
            text: The text to synthesize.
            language: Target language for the output speech.
            instruct: Optional emotion/style instruction (e.g. "Speak warmly").
            temperature: Sampling temperature (default 0.9).
            top_k: Top-K filtering (default 50).
            top_p: Nucleus sampling threshold (default 1.0).

        Returns:
            WAV audio file.
        """
        endpoint = f"POST /api/voices/{voice_name}/generate"
        log_request(endpoint, {
            "voice": voice_name, "text": text, "language": language,
            "instruct": instruct, "volume": volume, "temperature": temperature,
            "top_k": top_k, "top_p": top_p,
        })
        t0 = time.time()
        try:
            require_model(MODEL_BASE, "base")
            audio_path, meta = get_voice_path_and_meta(voice_name)
            ref_text = meta.get("ref_text") or None

            kwargs = dict(
                text=text,
                language=language,
                ref_audio=audio_path,
                ref_text=ref_text,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            if instruct and instruct.strip():
                kwargs["instruct"] = instruct

            wavs, sr = MODEL_BASE.generate_voice_clone(**kwargs)
            path = save_wav(wavs[0], sr, volume)
            log_result(endpoint, time.time() - t0, path)
            return FileResponse(path, media_type="audio/wav", filename="output.wav")
        except FileNotFoundError:
            log_result(endpoint, time.time() - t0, error=f"Voice '{voice_name}' not found")
            return JSONResponse(status_code=404, content={"error": f"Voice '{voice_name}' not found."})
        except Exception as e:
            log_result(endpoint, time.time() - t0, error=str(e))
            return JSONResponse(status_code=500, content={"error": str(e)})

    @fast_app.delete("/api/voices/{voice_name}")
    async def api_delete_voice(voice_name: str):
        """Delete a saved voice.

        Parameters:
            voice_name: The name of the saved voice to delete.

        Returns:
            {"message": "Voice deleted."}
        """
        log.info(f"\033[1;36mAPI\033[0m \033[1mDELETE /api/voices/{voice_name}\033[0m")
        if delete_voice(voice_name):
            log.info(f"\033[1;32mDONE\033[0m DELETE /api/voices/{voice_name} -> deleted")
            return {"message": f"Voice '{voice_name}' deleted."}
        log.info(f"\033[1;31mERROR\033[0m DELETE /api/voices/{voice_name} -> not found")
        return JSONResponse(status_code=404, content={"error": f"Voice '{voice_name}' not found."})

    @fast_app.get("/api/speakers")
    async def api_speakers():
        """List available speakers for the Custom Voice mode."""
        log.info(f"\033[1;36mAPI\033[0m \033[1mGET /api/speakers\033[0m -> {len(SPEAKERS)} speakers")
        return {"speakers": SPEAKERS}

    @fast_app.get("/api/languages")
    async def api_languages():
        """List supported languages."""
        log.info(f"\033[1;36mAPI\033[0m \033[1mGET /api/languages\033[0m -> {len(LANGUAGES)} languages")
        return {"languages": LANGUAGES}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def detect_device() -> tuple[str, torch.dtype]:
    """Pick the best available device and a safe default dtype."""
    if torch.cuda.is_available():
        return "cuda:0", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3-TTS Gradio + API Server")
    parser.add_argument("--device", default="auto", help="Torch device: auto, cuda:0, mps, cpu (default: auto)")
    parser.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"], help="Model dtype (default: auto)")
    parser.add_argument("--models", default="all",
                        help="Which models to load: all, or comma-separated list of: custom, design, base (default: all)")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    return parser.parse_args()


def main():
    args = parse_args()

    auto_device, auto_dtype = detect_device()
    device = auto_device if args.device == "auto" else args.device

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = auto_dtype if args.dtype == "auto" else dtype_map[args.dtype]

    if args.models == "all":
        models = VALID_MODELS
    else:
        models = {m.strip() for m in args.models.split(",")}
        invalid = models - VALID_MODELS
        if invalid:
            print(f"Unknown model(s): {invalid}. Valid: {VALID_MODELS}")
            return

    VOICES_DIR.mkdir(exist_ok=True)

    log.info(f"Using device: {device}, dtype: {dtype}")
    load_models(device, dtype, models)

    gradio_app = build_gradio_app()

    # Mount REST API onto the underlying FastAPI app
    fast_app = FastAPI(
        title="Qwen3-TTS API",
        description="REST API for Qwen3-TTS text-to-speech. Supports custom voice, voice design, voice cloning, and saved voices.",
        version="1.0.0",
    )
    build_api(fast_app)

    # Mount Gradio onto FastAPI
    app = gr.mount_gradio_app(fast_app, gradio_app, path="/")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
