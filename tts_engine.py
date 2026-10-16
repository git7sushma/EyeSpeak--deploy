"""
More natural-sounding text-to-speech for EyeSpeak, using Microsoft Edge's
free neural voices (edge-tts) instead of gTTS's more robotic-sounding output.

Falls back to gTTS automatically if edge-tts fails for any reason (e.g. an
unavailable voice name, or no internet access to Microsoft's speech service)
so a demo never breaks because of this - it just sounds slightly more
robotic for that one attempt.
"""
import asyncio
import edge_tts
from gtts import gTTS

# Best-known neural voice per language. Verify/adjust these yourself by
# running check_voices.py once on your own machine (full internet access
# needed - this couldn't be verified from the sandbox this was built in).
VOICE_MAP = {
    'en': 'en-US-AriaNeural',
    'ar': 'ar-AE-FatimaNeural',   # UAE Arabic specifically
    'hi': 'hi-IN-SwaraNeural',
    'ur': 'ur-PK-UzmaNeural',
    'bn': 'bn-IN-TanishaaNeural',
    'ml': 'ml-IN-SobhanaNeural',
    'ta': 'ta-IN-PallaviNeural',
    'gu': 'gu-IN-DhwaniNeural',
    'tl': 'fil-PH-BlessicaNeural',  # edge-tts uses 'fil', not 'tl'
    'te': 'te-IN-ShrutiNeural',
}


async def _edge_tts_save(text, voice, output_path):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)


def text_to_speech(text, lang_code, output_path):
    """Save `text` as speech audio at `output_path`, in the given language."""
    voice = VOICE_MAP.get(lang_code)
    if voice:
        try:
            asyncio.run(_edge_tts_save(text, voice, output_path))
            return
        except Exception as e:
            print(f"edge-tts failed ({e}), falling back to gTTS for this request.")

    # Fallback: gTTS (more robotic, but always works if edge-tts can't)
    tts = gTTS(text, lang=lang_code)
    tts.save(output_path)
