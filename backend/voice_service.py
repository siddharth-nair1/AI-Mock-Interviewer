from faster_whisper import WhisperModel
import edge_tts
import os
import subprocess

print("Loading Whisper model...")
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
print("Whisper model loaded!")

def convert_webm_to_wav(webm_path, wav_path):
    try:
        subprocess.run(
            ['ffmpeg', '-i', webm_path, '-ar', '16000', '-ac', '1', '-y', wav_path],
            check=True, capture_output=True
        )
        return wav_path
    except:
        return webm_path

def transcribe_audio(audio_file_path):
    try:
        print(f"Transcribing: {audio_file_path}")
        
        if not os.path.exists(audio_file_path):
            raise Exception(f"Audio file not found")
        
        file_size = os.path.getsize(audio_file_path)
        print(f"File size: {file_size} bytes")
        
        if file_size < 1000:
            raise Exception("Audio file too small")
        
        if audio_file_path.endswith('.webm'):
            wav_path = audio_file_path.replace('.webm', '.wav')
            try:
                audio_file_path = convert_webm_to_wav(audio_file_path, wav_path)
            except:
                pass
        
        segments, info = whisper_model.transcribe(
            audio_file_path, beam_size=5, language='en', vad_filter=True
        )
        
        text = " ".join(segment.text for segment in segments).strip()
        print(f"Transcription: '{text}'")
        
        if not text or len(text) < 3:
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
        
        # Add slight pauses for punctuation (Edge TTS handles this well, but we ensure cleanliness)
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