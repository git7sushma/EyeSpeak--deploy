"""
Run this once on your own machine (needs normal internet access) to confirm
the voice names in tts_engine.py's VOICE_MAP actually exist, and to see
alternatives if you want a different-sounding voice for any language.

Usage: python check_voices.py
"""
import asyncio
import edge_tts

LANGS_OF_INTEREST = ['en', 'ar', 'hi', 'ur', 'bn', 'ml', 'ta', 'gu', 'tl', 'fil', 'te']


async def main():
    voices = await edge_tts.list_voices()
    for lang in LANGS_OF_INTEREST:
        matches = [v for v in voices if v['Locale'].lower().startswith(lang)]
        print(f"\n--- {lang} ---")
        if not matches:
            print("  (no voices found for this code)")
        for v in matches:
            print(f"  {v['ShortName']}  ({v['Gender']})")


if __name__ == '__main__':
    asyncio.run(main())
