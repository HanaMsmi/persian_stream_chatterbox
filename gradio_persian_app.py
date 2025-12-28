"""
Persian Streaming TTS - Gradio Web Application
A beautiful web interface for multilingual text-to-speech with streaming support.
"""

import random
import numpy as np
import torch
import gradio as gr
from chatterbox import ChatterboxMultilingualStreamingTTS, SUPPORTED_LANGUAGES

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Running on device: {DEVICE}")

# --- Global Model Initialization ---
MODEL = None

# Language-specific example texts
LANGUAGE_CONFIG = {
    "fa": {
        "text": "سلام، خوش آمدید به دنیای هوش مصنوعی و تبدیل متن به گفتار. این یک نمونه متن فارسی است.",
        "name": "Persian (فارسی)",
    },
    "ar": {
        "text": "مرحبا، أهلا بكم في عالم الذكاء الاصطناعي وتحويل النص إلى كلام.",
        "name": "Arabic (العربية)",
    },
    "en": {
        "text": "Hello, welcome to the world of artificial intelligence and text-to-speech technology.",
        "name": "English",
    },
    "fr": {
        "text": "Bonjour, bienvenue dans le monde de l'intelligence artificielle et de la synthèse vocale.",
        "name": "French (Français)",
    },
    "de": {
        "text": "Hallo, willkommen in der Welt der künstlichen Intelligenz und der Text-zu-Sprache-Technologie.",
        "name": "German (Deutsch)",
    },
    "es": {
        "text": "Hola, bienvenidos al mundo de la inteligencia artificial y la tecnología de texto a voz.",
        "name": "Spanish (Español)",
    },
    "tr": {
        "text": "Merhaba, yapay zeka ve metin okuma teknolojisi dünyasına hoş geldiniz.",
        "name": "Turkish (Türkçe)",
    },
    "ru": {
        "text": "Привет, добро пожаловать в мир искусственного интеллекта и технологии преобразования текста в речь.",
        "name": "Russian (Русский)",
    },
    "zh": {
        "text": "你好，欢迎来到人工智能和文本转语音技术的世界。",
        "name": "Chinese (中文)",
    },
    "ja": {
        "text": "こんにちは、人工知能とテキスト読み上げ技術の世界へようこそ。",
        "name": "Japanese (日本語)",
    },
    "ko": {
        "text": "안녕하세요, 인공지능과 텍스트 음성 변환 기술의 세계에 오신 것을 환영합니다.",
        "name": "Korean (한국어)",
    },
    "hi": {
        "text": "नमस्ते, कृत्रिम बुद्धिमत्ता और टेक्स्ट-टू-स्पीच तकनीक की दुनिया में आपका स्वागत है।",
        "name": "Hindi (हिंदी)",
    },
}


def get_or_load_model():
    """Loads the ChatterboxMultilingualStreamingTTS model if it hasn't been loaded already."""
    global MODEL
    if MODEL is None:
        print("📥 Model not loaded, initializing...")
        try:
            MODEL = ChatterboxMultilingualStreamingTTS.from_pretrained(DEVICE)
            print(f"✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    return MODEL


def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_language_choices():
    """Get list of language choices for dropdown."""
    choices = []
    for code in LANGUAGE_CONFIG.keys():
        name = LANGUAGE_CONFIG[code]["name"]
        choices.append((name, code))
    # Add remaining supported languages
    for code, name in SUPPORTED_LANGUAGES.items():
        if code not in LANGUAGE_CONFIG:
            choices.append((name, code))
    return choices


def default_text_for_ui(lang: str) -> str:
    """Get default text for a language."""
    return LANGUAGE_CONFIG.get(lang, {}).get(
        "text", "Hello, this is a test of the text-to-speech system."
    )


def generate_tts_audio(
    text_input: str,
    language_id: str,
    audio_prompt_path: str = None,
    exaggeration_input: float = 0.5,
    temperature_input: float = 0.8,
    seed_num_input: int = 0,
    cfg_weight_input: float = 0.5,
    use_streaming: bool = True,
) -> tuple[int, np.ndarray]:
    """
    Generate speech audio from text using the multilingual streaming model.
    """
    current_model = get_or_load_model()

    if current_model is None:
        raise RuntimeError("TTS model is not loaded.")

    if seed_num_input != 0:
        set_seed(int(seed_num_input))

    print(f"📝 Generating audio for: '{text_input[:50]}...'")
    print(f"🌍 Language: {language_id}")

    generate_kwargs = {
        "exaggeration": exaggeration_input,
        "temperature": temperature_input,
        "cfg_weight": cfg_weight_input,
    }

    if audio_prompt_path and str(audio_prompt_path).strip():
        generate_kwargs["audio_prompt_path"] = audio_prompt_path
        print(f"🎤 Using audio prompt: {audio_prompt_path}")

    if use_streaming:
        # Streaming generation
        streamed_chunks = []
        for audio_chunk, metrics in current_model.generate_stream(
            text=text_input[:500],  # Limit text length
            language_id=language_id,
            chunk_size=25,
            print_metrics=False,
            **generate_kwargs,
        ):
            streamed_chunks.append(audio_chunk)

        if streamed_chunks:
            wav = torch.cat(streamed_chunks, dim=-1)
        else:
            raise RuntimeError("No audio chunks generated")
    else:
        # Non-streaming generation
        wav = current_model.generate(
            text=text_input[:500], language_id=language_id, **generate_kwargs
        )

    print("✅ Audio generation complete.")
    return (current_model.sr, wav.squeeze(0).numpy())


def on_language_change(lang):
    """Handle language change event."""
    return default_text_for_ui(lang)


# --- Build Gradio Interface ---
def create_demo():
    with gr.Blocks(
        title="Persian Streaming TTS",
        theme=gr.themes.Soft(
            primary_hue="emerald",
            secondary_hue="teal",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Vazirmatn"),
        ),
        css="""
        .rtl-text textarea { direction: rtl; text-align: right; font-family: 'Vazirmatn', 'Tahoma', sans-serif; }
        .container { max-width: 900px; margin: auto; }
        .header { text-align: center; margin-bottom: 20px; }
        .footer { text-align: center; margin-top: 30px; opacity: 0.7; }
        """,
    ) as demo:
        gr.Markdown(
            """
            <div class="header">
            
            # 🎙️ Persian Streaming TTS
            
            ### چترباکس فارسی - تبدیل متن به گفتار با قابلیت استریمینگ
            
            Generate high-quality multilingual speech with real-time streaming support.
            Supports **24 languages** including Persian, Arabic, English, French, and more.
            
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                initial_lang = "fa"

                language_id = gr.Dropdown(
                    choices=get_language_choices(),
                    value=initial_lang,
                    label="🌍 Language / زبان",
                    info="Select the language for synthesis",
                )

                text = gr.Textbox(
                    value=default_text_for_ui(initial_lang),
                    label="📝 Text to synthesize / متن برای تبدیل (max 500 chars)",
                    max_lines=5,
                    lines=3,
                    rtl=True,
                    elem_classes=["rtl-text"],
                )

                ref_wav = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="🎤 Reference Audio (Optional) / صدای مرجع",
                )

                gr.Markdown(
                    """
                    💡 **نکته**: برای کلون کردن صدا، یک فایل صوتی ۵ تا ۱۰ ثانیه‌ای آپلود کنید.
                    
                    💡 **Tip**: Upload a 5-10 second audio clip for voice cloning.
                    """
                )

            with gr.Column(scale=1):
                exaggeration = gr.Slider(
                    0.25,
                    2,
                    step=0.05,
                    value=0.5,
                    label="😊 Exaggeration / شدت احساس",
                    info="Higher = more expressive",
                )

                cfg_weight = gr.Slider(
                    0,
                    1,
                    step=0.05,
                    value=0.5,
                    label="🎯 CFG Weight / وزن راهنما",
                    info="Set to 0 for cross-language cloning",
                )

                with gr.Accordion("⚙️ Advanced Options", open=False):
                    seed_num = gr.Number(value=0, label="🎲 Random Seed (0 = random)")
                    temp = gr.Slider(
                        0.05, 5, step=0.05, value=0.8, label="🌡️ Temperature"
                    )
                    use_streaming = gr.Checkbox(
                        value=True, label="⚡ Use Streaming Generation"
                    )

        run_btn = gr.Button("🎵 Generate / تولید صدا", variant="primary", size="lg")

        audio_output = gr.Audio(label="🔊 Output Audio / خروجی صوتی")

        # Event handlers
        language_id.change(
            fn=on_language_change,
            inputs=[language_id],
            outputs=[text],
            show_progress=False,
        )

        run_btn.click(
            fn=generate_tts_audio,
            inputs=[
                text,
                language_id,
                ref_wav,
                exaggeration,
                temp,
                seed_num,
                cfg_weight,
                use_streaming,
            ],
            outputs=[audio_output],
        )

        gr.Markdown(
            """
            <div class="footer">
            
            ---
            
            🌟 **Persian Streaming Chatterbox** - Built with ❤️ for Persian speakers
            
            Based on [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) by Resemble AI
            
            </div>
            """
        )

    return demo


# Try to load model on startup
try:
    get_or_load_model()
except Exception as e:
    print(f"⚠️ Warning: Could not load model on startup: {e}")

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(share=True)
