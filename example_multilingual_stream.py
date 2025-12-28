"""
Multilingual Streaming TTS Example
Demonstrates streaming TTS for multiple languages including Persian, Arabic, French, etc.
"""
import torchaudio as ta
import torch
from chatterbox import ChatterboxMultilingualStreamingTTS, SUPPORTED_LANGUAGES

# Example texts for different languages
LANGUAGE_SAMPLES = {
    "fa": "سلام، من یک مدل تبدیل متن به گفتار هستم که از زبان فارسی پشتیبانی می‌کند.",
    "ar": "مرحبا، أنا نموذج تحويل النص إلى كلام يدعم اللغة العربية.",
    "en": "Hello, I am a text-to-speech model that supports the English language.",
    "fr": "Bonjour, je suis un modèle de synthèse vocale qui prend en charge la langue française.",
    "de": "Hallo, ich bin ein Text-to-Speech-Modell, das die deutsche Sprache unterstützt.",
    "es": "Hola, soy un modelo de texto a voz que admite el idioma español.",
    "zh": "你好，我是一个支持中文的文本转语音模型。",
    "ja": "こんにちは、私は日本語をサポートするテキスト読み上げモデルです。",
    "ko": "안녕하세요, 저는 한국어를 지원하는 텍스트 음성 변환 모델입니다.",
    "ru": "Привет, я модель преобразования текста в речь, поддерживающий русский язык.",
    "tr": "Merhaba, ben Türkçe dilini destekleyen bir metin okuma modeliyim.",
    "hi": "नमस्ते, मैं एक टेक्स्ट-टू-स्पीच मॉडल हूं जो हिंदी भाषा का समर्थन करता है।",
}


def main():
    # Automatically detect the best available device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"🚀 Using device: {device}")
    print("\n🌍 Supported Languages:")
    for code, name in sorted(SUPPORTED_LANGUAGES.items()):
        print(f"   {code}: {name}")
    
    print("\n📥 Loading multilingual streaming model...")
    model = ChatterboxMultilingualStreamingTTS.from_pretrained(device=device)
    print("✅ Model loaded successfully!")

    # Select languages to generate
    languages_to_test = ["fa", "ar", "en", "fr"]  # Persian, Arabic, English, French
    
    for lang_id in languages_to_test:
        if lang_id not in LANGUAGE_SAMPLES:
            print(f"⚠️ No sample text for language: {lang_id}")
            continue
            
        text = LANGUAGE_SAMPLES[lang_id]
        lang_name = SUPPORTED_LANGUAGES.get(lang_id, lang_id)
        
        print(f"\n{'='*60}")
        print(f"🌐 Generating: {lang_name} ({lang_id})")
        print(f"📝 Text: {text[:50]}..." if len(text) > 50 else f"📝 Text: {text}")
        
        # Streaming generation
        streamed_chunks = []
        try:
            for audio_chunk, metrics in model.generate_stream(
                text=text,
                language_id=lang_id,
                chunk_size=25,
                exaggeration=0.5,
                temperature=0.8,
                cfg_weight=0.5,
                print_metrics=False  # Quiet mode for batch processing
            ):
                streamed_chunks.append(audio_chunk)
                
        except Exception as e:
            print(f"❌ Error generating {lang_name}: {e}")
            continue

        # Save audio
        if streamed_chunks:
            full_audio = torch.cat(streamed_chunks, dim=-1)
            output_file = f"output_{lang_id}.wav"
            ta.save(output_file, full_audio, model.sr)
            duration = full_audio.shape[-1] / model.sr
            print(f"✅ Saved: {output_file} ({duration:.2f}s, {len(streamed_chunks)} chunks)")

    print(f"\n{'='*60}")
    print("🎉 MULTILINGUAL STREAMING DEMO COMPLETE!")


if __name__ == "__main__":
    main()

