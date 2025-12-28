"""
Voice Cloning with Streaming TTS Example
Demonstrates how to clone a voice from a reference audio file with streaming generation.
"""
import torchaudio as ta
import torch
from chatterbox import ChatterboxMultilingualStreamingTTS


def main():
    # Automatically detect the best available device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"🚀 Using device: {device}")
    print("📥 Loading multilingual streaming model...")
    
    model = ChatterboxMultilingualStreamingTTS.from_pretrained(device=device)
    print("✅ Model loaded successfully!")

    # === Configuration ===
    # Set path to your reference audio file (5-10 seconds of speech recommended)
    REFERENCE_AUDIO_PATH = "reference_voice.wav"  # Change this to your audio file
    
    # Text to synthesize (Persian example)
    text = "سلام، من صدای شما را تقلید می‌کنم و این متن را با همان لحن و سبک می‌خوانم."
    language_id = "fa"
    
    print(f"\n📝 Text: {text}")
    print(f"🌍 Language: Persian (fa)")
    print(f"🎤 Reference audio: {REFERENCE_AUDIO_PATH}")
    
    # Check if reference audio exists
    import os
    if not os.path.exists(REFERENCE_AUDIO_PATH):
        print(f"\n⚠️ Reference audio file not found: {REFERENCE_AUDIO_PATH}")
        print("Please provide a reference audio file (5-10 seconds of speech)")
        print("\nUsing default voice instead...")
        audio_prompt = None
    else:
        audio_prompt = REFERENCE_AUDIO_PATH
    
    # ==========================================
    # Streaming generation with voice cloning
    # ==========================================
    print("\n" + "="*60)
    print("📢 Generating audio with voice cloning (streaming)...")
    
    streamed_chunks = []
    
    try:
        for audio_chunk, metrics in model.generate_stream(
            text=text,
            language_id=language_id,
            audio_prompt_path=audio_prompt,
            chunk_size=25,
            exaggeration=0.5,  # Lower for more natural, higher for expressive
            temperature=0.8,
            cfg_weight=0.5,  # Set to 0 for better accent matching
            print_metrics=True
        ):
            streamed_chunks.append(audio_chunk)
            chunk_duration = audio_chunk.shape[-1] / model.sr
            print(f"📦 Chunk {metrics.chunk_count}: {chunk_duration:.3f}s")

    except Exception as e:
        print(f"❌ Error during generation: {e}")
        return

    # Save the complete audio
    if streamed_chunks:
        full_audio = torch.cat(streamed_chunks, dim=-1)
        output_file = "cloned_voice_output.wav"
        ta.save(output_file, full_audio, model.sr)
        
        print(f"\n✅ Saved: {output_file}")
        print(f"   Duration: {full_audio.shape[-1] / model.sr:.2f}s")
        print(f"   Chunks: {len(streamed_chunks)}")
    else:
        print("❌ No audio chunks were generated!")

    # ==========================================
    # Tips for better voice cloning
    # ==========================================
    print("\n" + "="*60)
    print("💡 Tips for better voice cloning:")
    print("   • Use 5-10 seconds of clean reference audio")
    print("   • Reference should be clear speech without background noise")
    print("   • Match reference language with target language for best results")
    print("   • Set cfg_weight=0 for cross-language voice cloning")
    print("   • Adjust exaggeration for expressiveness (0.3-0.7 recommended)")
    
    print("\n🎉 VOICE CLONING DEMO COMPLETE!")


if __name__ == "__main__":
    main()

