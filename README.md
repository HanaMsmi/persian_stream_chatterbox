# 🎙️ Persian Streaming Chatterbox TTS

<div dir="rtl">

## چترباکس فارسی - تبدیل متن به گفتار با قابلیت استریمینگ

این پروژه ترکیبی از مدل چندزبانه Chatterbox (با پشتیبانی از ۲۴ زبان از جمله فارسی) و قابلیت تولید صدا به صورت استریمینگ است.

</div>

---

## Features

- 🌍 **24 Languages Supported** - Including Persian (Farsi), Arabic, English, French, German, and more
- ⚡ **Real-time Streaming** - Generate audio chunks as the model produces them
- 🎤 **Voice Cloning** - Clone any voice from a reference audio file
- 😊 **Emotion Control** - Adjust expressiveness with the exaggeration parameter
- 🎯 **CFG Guidance** - Control generation quality with classifier-free guidance
- 🖥️ **Gradio Web UI** - Beautiful web interface for easy interaction

## Supported Languages

| Code | Language | Code | Language |
|------|----------|------|----------|
| fa | Persian (فارسی) | ar | Arabic (العربية) |
| en | English | fr | French |
| de | German | es | Spanish |
| tr | Turkish | ru | Russian |
| zh | Chinese | ja | Japanese |
| ko | Korean | hi | Hindi |
| it | Italian | pt | Portuguese |
| nl | Dutch | pl | Polish |
| ... | and more! | | |

## Installation

```bash
# Clone the repository
cd persian_stream_chatterbox

# Install with pip
pip install -e .

# Or install dependencies manually
pip install torch torchaudio transformers safetensors librosa gradio sounddevice
```

## Quick Start

### Basic Persian TTS (Streaming)

```python
import torch
import torchaudio as ta
from chatterbox import ChatterboxMultilingualStreamingTTS

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ChatterboxMultilingualStreamingTTS.from_pretrained(device=device)

# Persian text
text = "سلام، خوش آمدید به دنیای هوش مصنوعی."

# Streaming generation
chunks = []
for audio_chunk, metrics in model.generate_stream(
    text=text,
    language_id="fa",  # Persian
    chunk_size=25,
):
    chunks.append(audio_chunk)
    print(f"Chunk {metrics.chunk_count}: {audio_chunk.shape[-1] / model.sr:.2f}s")

# Save audio
full_audio = torch.cat(chunks, dim=-1)
ta.save("persian_output.wav", full_audio, model.sr)
```

### Non-Streaming Generation

```python
# Simple generation without streaming
wav = model.generate(
    text="سلام، این یک نمونه متن فارسی است.",
    language_id="fa",
    exaggeration=0.5,
    cfg_weight=0.5,
)
ta.save("output.wav", wav, model.sr)
```

### Voice Cloning

```python
# Clone voice from reference audio
for audio_chunk, metrics in model.generate_stream(
    text="متن فارسی با صدای کلون شده",
    language_id="fa",
    audio_prompt_path="reference_voice.wav",
    exaggeration=0.5,
    cfg_weight=0.0,  # Set to 0 for cross-language cloning
):
    # Process chunks...
    pass
```

## Web Interface

Launch the Gradio web interface:

```bash
python gradio_persian_app.py
```

This will start a beautiful web UI where you can:
- Select from 24 languages
- Type or paste text
- Upload reference audio for voice cloning
- Adjust generation parameters
- Generate and download audio

## Example Scripts

| Script | Description |
|--------|-------------|
| `example_persian_stream.py` | Persian streaming TTS with real-time playback |
| `example_multilingual_stream.py` | Generate audio for multiple languages |
| `example_voice_cloning_stream.py` | Voice cloning with streaming |
| `gradio_persian_app.py` | Web interface application |

## Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `exaggeration` | 0.25-2.0 | 0.5 | Emotion intensity (higher = more expressive) |
| `cfg_weight` | 0.0-1.0 | 0.5 | Guidance strength (0 for cross-language cloning) |
| `temperature` | 0.05-5.0 | 0.8 | Sampling randomness |
| `chunk_size` | 10-100 | 25 | Tokens per streaming chunk |

## Tips

### General Usage
- Default settings (`exaggeration=0.5`, `cfg_weight=0.5`) work well for most cases
- For cross-language voice cloning, set `cfg_weight=0` to avoid accent transfer

### Expressive Speech
- Try `exaggeration=0.7` and `cfg_weight=0.3` for more dramatic speech
- Higher exaggeration speeds up speech; lower CFG compensates with slower pacing

### Voice Cloning
- Use 5-10 seconds of clean reference audio
- Match reference language with target language for best results
- Set `cfg_weight=0` for cross-language cloning

## Model Architecture

This project combines:
1. **Multilingual T3 Model** (`t3_mtl23ls_v2.safetensors`) - 500M parameter LLaMA backbone
2. **S3Gen Vocoder** - High-fidelity audio synthesis
3. **Voice Encoder** - Speaker embedding extraction
4. **Streaming Generation** - Token-by-token audio chunk generation

## Credits

- **Chatterbox TTS** by [Resemble AI](https://resemble.ai) - Original multilingual model
- **Chatterbox-Streaming** by [David Browne](https://github.com/davidbrowne17) - Streaming implementation

## License

MIT License - See LICENSE file for details.

---

<div dir="rtl">

## مستندات فارسی

### نصب و راه‌اندازی

۱. پایتون ۳.۱۰ یا بالاتر نصب کنید
۲. کتابخانه‌های مورد نیاز را نصب کنید:
```bash
pip install -e .
```

### استفاده سریع

```python
from chatterbox import ChatterboxMultilingualStreamingTTS

model = ChatterboxMultilingualStreamingTTS.from_pretrained("cuda")

for chunk, metrics in model.generate_stream(
    text="سلام، این یک تست است.",
    language_id="fa",
):
    # پردازش هر تکه صوتی
    pass
```

</div>

