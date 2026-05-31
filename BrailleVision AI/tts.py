# tts.py

import pyttsx3


def speak(text, rate=135, volume=1.0):
    """Speak text on the local machine and return a success flag."""

    if not text or not text.strip():
        return False, "No text to speak."

    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", rate)
        engine.setProperty("volume", max(0.0, min(float(volume), 1.0)))
        engine.say(text)
        engine.runAndWait()
        engine.stop()
        return True, None
    except Exception as exc:
        return False, str(exc)
