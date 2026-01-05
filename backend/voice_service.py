from faster_whisper import WhisperModel
import edge_tts
import os

print("Loading Whisper model...")
# Optimized for Speed: CUDA + float16
try:
    whisper_model = WhisperModel("base", device="cuda", compute_type="float16")
    print("Whisper model loaded on GPU (CUDA)!")
except Exception as e:
    print(f"Warning: GPU init failed ({e}). Falling back to CPU.")
    whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

def transcribe_audio(audio_file_path):
    try:
        print(f"Transcribing: {audio_file_path}")
        
        if not os.path.exists(audio_file_path):
            raise Exception(f"Audio file not found")
        
        file_size = os.path.getsize(audio_file_path)
        print(f"File size: {file_size} bytes")
        
        if file_size < 100:
            raise Exception("Audio file too small")
        
        # faster-whisper handles WebM/MP3/WAV natively via internal ffmpeg libraries
        segments, info = whisper_model.transcribe(
            audio_file_path, 
            beam_size=1,      # Greedy search for speed
            language='en', 
            vad_filter=True   # Filters out silence/noise
        )
        
        text = " ".join(segment.text for segment in segments).strip()
        print(f"Transcription: '{text}'")
        
        if not text:
            raise Exception("No speech detected")
        
        return text
    except Exception as e:
        raise Exception(f"Transcription error: {e}")

async def text_to_speech(text, output_path):
    """Edge-TTS with async support for FastAPI"""
    try:
        print(f"Generating speech: {text[:80]}...")
        
        # Clean text
        text = ' '.join(text.replace('\n', ' ').replace('\r', ' ').split())
        
        # Add slight pauses for punctuation
        text = text.replace('.', '. ').replace(',', ', ').replace('?', '? ').replace('!', '! ')
        
        if len(text) > 500:
            text = text[:497] + "..."
        
        # Choose voice
        voice = "en-US-ChristopherNeural"
        
        # Generate speech
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_path)
        
        if not os.path.exists(output_path):
            raise Exception("Audio not generated")
        
        print(f"✅ Speech generated: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"TTS error: {e}")
        raise Exception(f"TTS error: {e}")
