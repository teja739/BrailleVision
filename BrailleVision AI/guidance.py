# guidance.py

import cv2
import numpy as np


def analyze_scan_quality(image, dots, translation, debug):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    text = translation.get("text", "")
    confidence = float(translation.get("confidence", 0.0))
    cell_count = len(translation.get("cells", []))

    tips = []
    level = "good"

    if brightness < 75:
        tips.append("Increase lighting. The image is too dark.")
        level = "warn"
    elif brightness > 225:
        tips.append("Reduce glare. The image is too bright.")
        level = "warn"

    if contrast < 18:
        tips.append("Move closer or improve contrast between dots and paper.")
        level = "warn"

    if len(dots) < 3:
        tips.append("Move closer and keep the Braille area centered.")
        level = "bad"
    elif cell_count == 0:
        tips.append("Dots were found, but not enough structure was found for Braille cells.")
        level = "bad"

    if "?" in text:
        tips.append("Some cells are unclear. Retake the photo straight-on.")
        level = "warn"

    if confidence >= 0.9 and text:
        summary = "Ready to read aloud."
    elif text:
        summary = "Readable, but review the text before speaking."
    else:
        summary = "No readable Braille yet."

    if not tips:
        tips.append("Image quality looks good for this demo pipeline.")

    spoken = f"{summary} Detected text is {text}." if text else summary

    return {
        "level": level,
        "summary": summary,
        "tips": tips,
        "spoken": spoken,
        "brightness": brightness,
        "contrast": contrast,
        "polarity": debug.get("polarity", "unknown"),
    }
