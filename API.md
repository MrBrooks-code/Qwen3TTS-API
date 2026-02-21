# Qwen3-TTS API Reference

Base URL: `http://localhost:7860`

Interactive Swagger docs available at: `http://localhost:7860/docs`

All audio endpoints return a `audio/wav` binary response. Save the response body directly to a `.wav` file to play it.

On error, endpoints return JSON: `{"error": "<message>"}` with HTTP status 500.

---

## Endpoints

### GET /api/speakers

List the available built-in speakers for the Custom Voice endpoint.

**Parameters:** None

**Response:**
```json
{
  "speakers": [
    "Vivian",
    "Serena",
    "Uncle_Fu",
    "Dylan",
    "Eric",
    "Ryan",
    "Aiden",
    "Ono_Anna",
    "Sohee"
  ]
}
```

**Speaker details:**

| Speaker   | Description                              | Best Language |
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

All speakers can synthesize any of the 10 supported languages, but they sound most natural in their best language.

---

### GET /api/languages

List supported languages.

**Parameters:** None

**Response:**
```json
{
  "languages": [
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "German",
    "French",
    "Russian",
    "Portuguese",
    "Spanish",
    "Italian"
  ]
}
```

---

### POST /api/custom-voice

Generate speech using one of the 9 built-in premium voices. Optionally control emotion, tone, and style via a natural language instruction.

**Content-Type:** `multipart/form-data`

**Parameters:**

| Field       | Type   | Required | Default   | Description |
|-------------|--------|----------|-----------|-------------|
| text        | string | yes      | —         | The text to synthesize into speech. |
| language    | string | no       | "English" | The language of the text. Must be one of the supported languages (see GET /api/languages). |
| speaker     | string | no       | "Ryan"    | The voice to use. Must be one of the supported speakers (see GET /api/speakers). |
| instruct    | string | no       | ""        | Natural language instruction to control speaking style, emotion, and tone. Leave empty for the speaker's default style. Examples: "Speak warmly and cheerfully", "Read this in a serious, professional tone", "Very excited and energetic". |
| temperature | float  | no       | 0.9       | Sampling temperature. Higher values (e.g. 1.2) produce more varied output. Lower values (e.g. 0.5) produce more predictable output. Range: 0.1–2.0. |
| top_k       | int    | no       | 50        | Top-K sampling. Limits token selection to the K most probable tokens. Range: 1–200. |
| top_p       | float  | no       | 1.0       | Nucleus sampling threshold. Range: 0.0–1.0. |

**Response:** Binary WAV audio (`audio/wav`).

**Example (curl):**
```bash
curl -X POST http://localhost:7860/api/custom-voice \
  -F "text=Hello! Welcome to the demo." \
  -F "language=English" \
  -F "speaker=Ryan" \
  -F "instruct=Speak warmly and cheerfully" \
  --output output.wav
```

**Example (Python):**
```python
import requests

response = requests.post("http://localhost:7860/api/custom-voice", data={
    "text": "Hello! Welcome to the demo.",
    "language": "English",
    "speaker": "Ryan",
    "instruct": "Speak warmly and cheerfully",
})
with open("output.wav", "wb") as f:
    f.write(response.content)
```

**When to use this endpoint:** Use this when you want a consistent, high-quality voice and you know which speaker to use. Best for conversational output, narration, or any scenario where you want a reliable voice with optional emotional styling.

---

### POST /api/voice-design

Generate speech with a completely new voice created from a natural language description. The model invents a voice that matches your description.

**Content-Type:** `multipart/form-data`

**Parameters:**

| Field       | Type   | Required | Default   | Description |
|-------------|--------|----------|-----------|-------------|
| text        | string | yes      | —         | The text to synthesize into speech. |
| language    | string | no       | "English" | The language of the text. Must be one of the supported languages. |
| instruct    | string | yes      | —         | Natural language description of the voice to create. Describe the voice's character, age, gender, tone, and style. Examples: "A deep, warm male voice with slight raspiness", "A bright, energetic young woman with a playful tone", "An elderly British gentleman speaking softly". |
| temperature | float  | no       | 0.9       | Sampling temperature. Range: 0.1–2.0. |
| top_k       | int    | no       | 50        | Top-K sampling. Range: 1–200. |
| top_p       | float  | no       | 1.0       | Nucleus sampling threshold. Range: 0.0–1.0. |

**Response:** Binary WAV audio (`audio/wav`).

**Example (curl):**
```bash
curl -X POST http://localhost:7860/api/voice-design \
  -F "text=The quick brown fox jumps over the lazy dog." \
  -F "language=English" \
  -F "instruct=A deep, warm male voice with slight raspiness" \
  --output designed_voice.wav
```

**Example (Python):**
```python
import requests

response = requests.post("http://localhost:7860/api/voice-design", data={
    "text": "The quick brown fox jumps over the lazy dog.",
    "language": "English",
    "instruct": "A deep, warm male voice with slight raspiness",
})
with open("designed_voice.wav", "wb") as f:
    f.write(response.content)
```

**When to use this endpoint:** Use this when none of the built-in speakers match what you want. You describe the voice you're imagining and the model creates it. Note: the generated voice may vary between calls since it's synthesized on the fly.

---

### POST /api/voice-clone

Clone a voice from a reference audio clip and use it to synthesize new text. Requires uploading a short audio sample of the target voice.

**Content-Type:** `multipart/form-data`

**Parameters:**

| Field       | Type   | Required | Default   | Description |
|-------------|--------|----------|-----------|-------------|
| text        | string | yes      | —         | The text to synthesize into speech. |
| language    | string | no       | "English" | The language of the text. Must be one of the supported languages. |
| ref_audio   | file   | yes      | —         | Reference audio file containing the voice to clone. Accepts WAV, MP3, FLAC, OGG, etc. A clip of approximately 3 seconds is recommended. Longer clips work but don't significantly improve quality. |
| ref_text    | string | no       | ""        | Transcript of what is spoken in the reference audio. Providing this significantly improves cloning quality. Leave empty if unknown. |
| temperature | float  | no       | 0.9       | Sampling temperature. Range: 0.1–2.0. |
| top_k       | int    | no       | 50        | Top-K sampling. Range: 1–200. |
| top_p       | float  | no       | 1.0       | Nucleus sampling threshold. Range: 0.0–1.0. |

**Response:** Binary WAV audio (`audio/wav`).

**Example (curl):**
```bash
curl -X POST http://localhost:7860/api/voice-clone \
  -F "text=This is a test of voice cloning." \
  -F "language=English" \
  -F "ref_audio=@reference.wav" \
  -F "ref_text=Hello, my name is Alice and this is my voice." \
  --output cloned.wav
```

**Example (Python):**
```python
import requests

with open("reference.wav", "rb") as audio_file:
    response = requests.post("http://localhost:7860/api/voice-clone", data={
        "text": "This is a test of voice cloning.",
        "language": "English",
        "ref_text": "Hello, my name is Alice and this is my voice.",
    }, files={
        "ref_audio": ("reference.wav", audio_file, "audio/wav"),
    })
with open("cloned.wav", "wb") as f:
    f.write(response.content)
```

**When to use this endpoint:** Use this when you have a specific voice you want to replicate. You supply a short audio sample and the model produces new speech that sounds like that person. Best results come from clean, single-speaker reference audio with a provided transcript.

---

## Saved Voices

Save cloned voices for reuse without re-uploading reference audio each time.

### GET /api/voices

List all saved cloned voices.

**Parameters:** None

**Response:**
```json
{
  "voices": [
    {
      "id": "my_narrator",
      "name": "My Narrator",
      "ref_text": "Hello, this is a sample of my voice.",
      "language": "English"
    }
  ]
}
```

---

### POST /api/voices

Save a cloned voice for later reuse.

**Content-Type:** `multipart/form-data`

**Parameters:**

| Field     | Type   | Required | Default   | Description |
|-----------|--------|----------|-----------|-------------|
| name      | string | yes      | —         | A human-readable name for this voice (e.g. "My Narrator"). Used to reference it later. |
| language  | string | no       | "English" | The language spoken in the reference audio. |
| ref_text  | string | no       | ""        | Transcript of the reference audio. Providing this improves generation quality. |
| ref_audio | file   | yes      | —         | Reference audio file (WAV, MP3, etc.) — ~3 seconds recommended. |

**Response:**
```json
{
  "id": "My_Narrator",
  "name": "My Narrator",
  "message": "Voice saved."
}
```

**Example (curl):**
```bash
curl -X POST http://localhost:7860/api/voices \
  -F "name=My Narrator" \
  -F "language=English" \
  -F "ref_text=Hello, this is a sample of my voice." \
  -F "ref_audio=@reference.wav"
```

**Example (Python):**
```python
import requests

with open("reference.wav", "rb") as audio_file:
    response = requests.post("http://localhost:7860/api/voices", data={
        "name": "My Narrator",
        "language": "English",
        "ref_text": "Hello, this is a sample of my voice.",
    }, files={
        "ref_audio": ("reference.wav", audio_file, "audio/wav"),
    })
print(response.json())
```

---

### POST /api/voices/{voice_name}/generate

Generate speech using a previously saved cloned voice. No need to re-upload reference audio.

**Content-Type:** `multipart/form-data`

**Parameters:**

| Field       | Type   | Required | Default   | Description |
|-------------|--------|----------|-----------|-------------|
| text        | string | yes      | —         | The text to synthesize into speech. |
| language    | string | no       | "English" | The language of the output speech. |
| instruct    | string | no       | ""        | Optional emotion/style instruction (e.g. "Speak warmly and cheerfully", "Read in a serious tone"). |
| temperature | float  | no       | 0.9       | Sampling temperature. Range: 0.1–2.0. |
| top_k       | int    | no       | 50        | Top-K sampling. Range: 1–200. |
| top_p       | float  | no       | 1.0       | Nucleus sampling threshold. Range: 0.0–1.0. |

**Response:** Binary WAV audio (`audio/wav`).

**Example (curl):**
```bash
curl -X POST http://localhost:7860/api/voices/My_Narrator/generate \
  -F "text=This is spoken in my saved voice." \
  -F "language=English" \
  -F "instruct=Speak warmly and cheerfully" \
  --output output.wav
```

**Example (Python):**
```python
import requests

response = requests.post("http://localhost:7860/api/voices/My_Narrator/generate", data={
    "text": "This is spoken in my saved voice.",
    "language": "English",
    "instruct": "Speak warmly and cheerfully",
})
with open("output.wav", "wb") as f:
    f.write(response.content)
```

**When to use this endpoint:** Use this when you've already saved a cloned voice and want to generate speech with it repeatedly. This is the simplest endpoint for LLM agents — just a voice name and text, no file uploads needed.

---

### DELETE /api/voices/{voice_name}

Delete a saved voice.

**Parameters:** None (voice name is in the URL path).

**Response:**
```json
{"message": "Voice 'My Narrator' deleted."}
```

**Example (curl):**
```bash
curl -X DELETE http://localhost:7860/api/voices/My_Narrator
```

---

## LLM Tool Definitions

If you are an LLM agent and want to call this API as a tool, here are the tool definitions you need.

### Tool: tts_custom_voice

Use this tool to generate speech audio from text using a built-in voice. Best for producing speech with consistent, high-quality voices with optional emotional/style control.

```json
{
  "name": "tts_custom_voice",
  "description": "Generate speech from text using a built-in premium voice. Returns a WAV audio file. Use the 'instruct' parameter to control emotion and speaking style.",
  "parameters": {
    "type": "object",
    "required": ["text"],
    "properties": {
      "text": {
        "type": "string",
        "description": "The text to convert to speech."
      },
      "language": {
        "type": "string",
        "enum": ["Chinese", "English", "Japanese", "Korean", "German", "French", "Russian", "Portuguese", "Spanish", "Italian"],
        "default": "English",
        "description": "The language of the input text."
      },
      "speaker": {
        "type": "string",
        "enum": ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ryan", "Aiden", "Ono_Anna", "Sohee"],
        "default": "Ryan",
        "description": "Which built-in voice to use. Ryan/Aiden are English male voices. Vivian/Serena are Chinese female voices. Ono_Anna is Japanese female. Sohee is Korean female."
      },
      "instruct": {
        "type": "string",
        "default": "",
        "description": "Optional natural language instruction controlling emotion, tone, and style. Examples: 'Speak warmly', 'Read in a serious tone', 'Very excited and happy'."
      }
    }
  }
}
```

**Calling convention:** Send a POST to `/api/custom-voice` with form data. Save the binary response as a `.wav` file.

### Tool: tts_voice_design

Use this tool when you need a specific type of voice that isn't available in the built-in speakers. You describe the voice and the model creates it.

```json
{
  "name": "tts_voice_design",
  "description": "Generate speech with a new voice created from a natural language description. Returns a WAV audio file. The voice is invented to match your description.",
  "parameters": {
    "type": "object",
    "required": ["text", "instruct"],
    "properties": {
      "text": {
        "type": "string",
        "description": "The text to convert to speech."
      },
      "language": {
        "type": "string",
        "enum": ["Chinese", "English", "Japanese", "Korean", "German", "French", "Russian", "Portuguese", "Spanish", "Italian"],
        "default": "English",
        "description": "The language of the input text."
      },
      "instruct": {
        "type": "string",
        "description": "Natural language description of the desired voice. Describe gender, age, tone, personality, and style. Example: 'A deep, warm male voice with slight raspiness and a calm demeanor'."
      }
    }
  }
}
```

**Calling convention:** Send a POST to `/api/voice-design` with form data. Save the binary response as a `.wav` file.

### Tool: tts_voice_clone

Use this tool when you have a reference audio file of a specific person's voice and want to generate new speech in that voice.

```json
{
  "name": "tts_voice_clone",
  "description": "Clone a voice from a reference audio clip and generate new speech in that voice. Requires a short (~3 second) audio sample. Returns a WAV audio file.",
  "parameters": {
    "type": "object",
    "required": ["text", "ref_audio"],
    "properties": {
      "text": {
        "type": "string",
        "description": "The text to convert to speech."
      },
      "language": {
        "type": "string",
        "enum": ["Chinese", "English", "Japanese", "Korean", "German", "French", "Russian", "Portuguese", "Spanish", "Italian"],
        "default": "English",
        "description": "The language of the input text."
      },
      "ref_audio": {
        "type": "string",
        "description": "Path to the reference audio file containing the voice to clone."
      },
      "ref_text": {
        "type": "string",
        "default": "",
        "description": "Transcript of what is spoken in the reference audio. Providing this improves cloning quality."
      }
    }
  }
}
```

**Calling convention:** Send a POST to `/api/voice-clone` with multipart form data. The `ref_audio` field must be a file upload. Save the binary response as a `.wav` file.

### Tool: tts_saved_voice

Use this tool to generate speech using a previously saved cloned voice. This is the simplest option when a voice has already been saved — just provide the voice name and text, no file upload needed.

```json
{
  "name": "tts_saved_voice",
  "description": "Generate speech using a previously saved cloned voice. No file upload needed — just specify the voice name and text. Call GET /api/voices first to see available voices. Returns a WAV audio file.",
  "parameters": {
    "type": "object",
    "required": ["voice_name", "text"],
    "properties": {
      "voice_name": {
        "type": "string",
        "description": "The name of the saved voice (as returned by GET /api/voices). Use the 'id' field from the voice list."
      },
      "text": {
        "type": "string",
        "description": "The text to convert to speech."
      },
      "language": {
        "type": "string",
        "enum": ["Chinese", "English", "Japanese", "Korean", "German", "French", "Russian", "Portuguese", "Spanish", "Italian"],
        "default": "English",
        "description": "The language of the input text."
      },
      "instruct": {
        "type": "string",
        "default": "",
        "description": "Optional natural language instruction controlling emotion, tone, and style. Examples: 'Speak warmly', 'Read in a serious tone', 'Very excited and happy'."
      }
    }
  }
}
```

**Calling convention:** Send a POST to `/api/voices/{voice_name}/generate` with form data. Save the binary response as a `.wav` file.

### Tool: tts_save_voice

Use this tool to save a cloned voice for later reuse.

```json
{
  "name": "tts_save_voice",
  "description": "Save a reference audio clip as a named voice for later reuse. After saving, use tts_saved_voice to generate speech without re-uploading.",
  "parameters": {
    "type": "object",
    "required": ["name", "ref_audio"],
    "properties": {
      "name": {
        "type": "string",
        "description": "A human-readable name for this voice (e.g. 'My Narrator')."
      },
      "ref_audio": {
        "type": "string",
        "description": "Path to the reference audio file containing the voice to save."
      },
      "ref_text": {
        "type": "string",
        "default": "",
        "description": "Transcript of what is spoken in the reference audio. Improves quality."
      },
      "language": {
        "type": "string",
        "default": "English",
        "description": "The language spoken in the reference audio."
      }
    }
  }
}
```

**Calling convention:** Send a POST to `/api/voices` with multipart form data. The `ref_audio` field must be a file upload.

---

## Choosing the Right Endpoint

| Scenario | Endpoint | Why |
|----------|----------|-----|
| You want consistent, repeatable voice output | `/api/custom-voice` | Built-in speakers always sound the same. |
| You want to control emotion or speaking style | `/api/custom-voice` with `instruct` | The instruct parameter adjusts tone, emotion, and pacing. |
| You need a voice type not in the built-in list | `/api/voice-design` | Describe any voice and the model creates it. |
| You want to sound like a specific person | `/api/voice-clone` | Provide a sample of their voice. |
| You already saved a cloned voice | `/api/voices/{name}/generate` | No file upload needed, just voice name + text. |
| You need multilingual output | Any endpoint | All endpoints support 10 languages. Set `language` accordingly. |

## Notes

- All audio output is WAV format, mono channel.
- Generation typically takes 1-5 seconds depending on text length and GPU.
- The `temperature`, `top_k`, and `top_p` parameters are optional. The defaults work well for most cases. Only adjust them if output quality is poor or you want more variation.
- Maximum text length per request is approximately 500 characters for best quality. For longer text, split into sentences and make multiple requests.
